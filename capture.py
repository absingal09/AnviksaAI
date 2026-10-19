"""
capture.py — the ears (and mouth). Turns a recorded audio file into a
diarized transcript, a photographed prescription into text, and text into
spoken audio.

Four functions:
    transcribe(path)        -> dict            (saaras-v3, batch + diarized)
    read_image(path)        -> dict            (sarvam-vision / doc_ai)
    translate(text)         -> str             (sarvam-translate / mayura)
    speak(text, language)   -> WAV bytes       (bulbul-v3)

Every call is cached to cache/ keyed by a hash of its input, so the same
audio/image/text is never sent to Sarvam twice — vision is capped at
10-30 req/min on every tier per CLAUDE.md, that's the ceiling this project
will hit, not credits. 429s are retried with exponential backoff.

Built and verified against the ACTUAL installed sarvamai==0.1.30 client,
not its docs — those turned out to be wrong about several method names
(target_language_code vs language_code, wait_until_complete() not
existing on doc_ai jobs, get_file_results() not containing the transcript
for batch STT jobs either). The original SDK smoke tests and exploratory
scripts in Git history document how the batch STT flow was verified.
If you extend this file, verify new calls the same way: introspect the
real client/response objects (or write a throwaway script and read the
actual output) before trusting a docs page.
"""
import hashlib
import base64
import json
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from sarvamai import SarvamAI

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

_client = None


def client():
    global _client
    if _client is None:
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("SARVAM_API_KEY not set — check .env")
        _client = SarvamAI(api_subscription_key=key)
    return _client


def _hash(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:24]


def _with_retry(fn, *args, max_retries=5, **kwargs):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status == 429 and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------
def transcribe(path, language_code="hi-IN", num_speakers=2):
    """Audio file -> diarized transcript, via saaras-v3 BATCH speech-to-text
    (CLAUDE.md's actual spec: "saaras-v3 batch STT (diarized)").

    Returns a dict shaped like Sarvam's own per-file output JSON:
        {"transcript": "full text",
         "diarized_transcript": {"entries": [
             {"transcript": ..., "start_time_seconds": ..., "end_time_seconds": ...,
              "speaker_id": "0"}, ...]},
         "language_code": ..., ...}
    so extract.py can use entries[i]["speaker_id"] to tell patient from
    doctor turns apart, rather than guessing from grammatical person alone.
    This replaces the single-string return type used by the earlier smoke
    tests, retained in Git history. Those tests called the simpler
    speech_to_text.transcribe directly.

    The real, verified call sequence (none of it guessable from docs —
    get_file_results()/get_status() are pass/fail bookkeeping only; the
    transcript only ever shows up in a file download_outputs() writes to
    disk):
        create_job(...) -> job.upload_files(file_paths=[...]) -> job.start()
        -> job.wait_until_complete(timeout=...) -> job.download_outputs(
        output_dir=...) WRITES "<original_filename>.json" per input file
        into that directory -> read/parse that file for the transcript.
    """
    path = Path(path)
    audio_bytes = path.read_bytes()
    cpath = CACHE_DIR / f"stt_{_hash(audio_bytes, language_code, num_speakers)}.json"
    if cpath.exists():
        return json.loads(cpath.read_text(encoding="utf-8"))

    job = _with_retry(
        client().speech_to_text_job.create_job,
        model="saaras:v3", language_code=language_code,
        with_diarization=True, num_speakers=num_speakers,
    )
    job.upload_files(file_paths=[str(path)])
    job.start()
    job.wait_until_complete(timeout=180)

    if job.is_failed():
        raise RuntimeError(f"batch STT job {job.job_id} failed for {path.name}")

    with tempfile.TemporaryDirectory() as out_dir:
        job.download_outputs(output_dir=out_dir)
        result_path = Path(out_dir) / f"{path.name}.json"
        if not result_path.exists():
            # naming drifted from what was verified — fall back to
            # whatever landed in out_dir rather than fail outright
            candidates = list(Path(out_dir).glob("*.json"))
            if not candidates:
                raise RuntimeError(
                    f"batch STT job {job.job_id} completed but wrote no "
                    f"output file into {out_dir}")
            result_path = candidates[0]
        result = json.loads(result_path.read_text(encoding="utf-8"))

    cpath.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


# --------------------------------------------------------------------------
def read_image(path, language="en-IN"):
    """Photo of a prescription/blister strip -> extracted text, via
    sarvam-vision (doc_ai.digitise). Polls manually — get_status()/
    get_results() are the real methods on this SDK version, not the
    wait_until_complete()/get_file_results() the docs describe."""
    path = Path(path)
    img_bytes = path.read_bytes()
    cpath = CACHE_DIR / f"vision_{_hash(img_bytes, language)}.json"
    if cpath.exists():
        return json.loads(cpath.read_text(encoding="utf-8"))

    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "pdf": "application/pdf"}.get(path.suffix.lstrip(".").lower(),
                                          "application/octet-stream")
    with open(path, "rb") as f:
        job = _with_retry(
            client().doc_ai.digitise,
            file=[(path.name, f, mime)], language=language, output_format="md",
        )

    deadline = time.time() + 60
    status = job.status
    while status not in ("completed", "failed"):
        if time.time() > deadline:
            raise TimeoutError(f"doc_ai job {job.job_id} still '{status}' after 60s")
        time.sleep(2)
        status = client().doc_ai.get_status(job.job_id).status
    if status == "failed":
        raise RuntimeError(f"doc_ai job {job.job_id} failed")

    results = client().doc_ai.get_results(job.job_id)
    blocks = []
    for doc in results.documents:
        for page in doc.pages:
            for b in page.blocks:
                text = b["text"] if isinstance(b, dict) else b.text
                blocks.append(text)
    extracted = {"blocks": blocks, "text": "\n".join(blocks)}
    cpath.write_text(json.dumps(extracted, ensure_ascii=False), encoding="utf-8")
    return extracted


# --------------------------------------------------------------------------
def speak(text, language_code="hi-IN", speaker="shubh"):
    """Text -> WAV bytes, via bulbul-v3. Cached by (text, language, speaker).

    Returns bytes rather than writing a file: the only caller streams this
    straight to the browser, and a temp file on the way there was pure
    ceremony.

    THIS SPEAKS TO A PATIENT. Two rules the caller must hold to, neither
    enforceable here:

    1. Never speak a finding flagged by vocabulary.is_clinician_only(). Those
       questions are phrased for the doctor — "Are the neck veins distended?"
       read aloud to a patient is nonsense at best.
    2. Never speak raw English text in a non-English voice. The previous
       build shipped a dict with exactly ONE hardcoded Hindi sentence and
       fell through to speaking English in an English voice for every other
       question and every other language. It went unnoticed because the
       scripted demo pinned the question to that one id. Translate first,
       and show the translation on screen — an unverified translation asked
       aloud produces an answer that enters the findings pipeline as genuine
       evidence, so it has to be inspectable.
    """
    key_hash = _hash(text, language_code, speaker)
    cpath = CACHE_DIR / f"tts_{key_hash}.wav"
    if not cpath.exists():
        resp = _with_retry(
            client().text_to_speech.convert,
            text=text, language_code=language_code, speaker=speaker,
            model="bulbul:v3",
        )
        audios = getattr(resp, "audios", None) or resp["audios"]
        cpath.write_bytes(base64.b64decode(audios[0]))
    return cpath.read_bytes()


# --------------------------------------------------------------------------
def translate(text, source_language_code="hi-IN", target_language_code="en-IN"):
    """Text -> translated text, via Sarvam's text.translate (mayura/
    sarvam-translate). Cached by (text, source, target) same as the other
    calls here. Best-effort by convention of every caller of this function,
    not enforced here — returns the translated string or raises, same as
    speak()/transcribe(); it's the CALLER's job to decide a translation
    failure shouldn't fail the whole request (see main.py's usage)."""
    if not text.strip():
        return ""
    key_hash = _hash(text, source_language_code, target_language_code)
    cpath = CACHE_DIR / f"translate_{key_hash}.json"
    if cpath.exists():
        return json.loads(cpath.read_text(encoding="utf-8"))["translated_text"]

    resp = _with_retry(
        client().text.translate,
        input=text, source_language_code=source_language_code,
        target_language_code=target_language_code,
    )
    translated = getattr(resp, "translated_text", None) or resp["translated_text"]
    cpath.write_text(json.dumps({"translated_text": translated}, ensure_ascii=False),
                      encoding="utf-8")
    return translated


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("usage: python capture.py <audio-or-image-path>")
        sys.exit(1)
    p = Path(sys.argv[1])
    if p.suffix.lower() in (".wav", ".mp3", ".m4a", ".ogg", ".flac"):
        print(json.dumps(transcribe(p), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(read_image(p), ensure_ascii=False, indent=2))
