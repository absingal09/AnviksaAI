"""main.py — the wire. FastAPI: three POST endpoints, a redirect, and a static mount.

    POST /capture   multipart/form-data. STT or OCR, nothing else.
        Either one "audio" file (+ optional "language_code")
            -> {"transcript": str, "transcript_en": str|absent}
        or one-or-more "documents" files
            -> {"scanned_documents": [{"name", "chars", "text"}, ...]}

    POST /speak     application/json. {"finding": id, "language_code": str}
        -> the question translated into the patient's language, spoken by
        Bulbul, PLUS the translated text so it can be shown on screen and
        checked. Refuses clinician-only questions.

    POST /analyze   application/json. extract -> propose -> triage, over
        everything accumulated so far.
            {"transcript": str, "doc_text": str,
             "confirmed_exclusions": [str], "asked": [str]}
            -> the panel payload (see engine.assess)

    GET  /web/...   the single-file frontend.

KEEP THESE TWO SEPARATE. They look mergeable and are not. When the frontend
used one combined endpoint, every recording and every upload ran a full
extraction whose result was immediately thrown away, then ran a second one
from the panel refresh — double the Sarvam cost and double the latency on
every single action. That merged route (/assess) has been deleted rather
than left available, because leaving it available is how it comes back.

WHY /analyze RE-SENDS THE WHOLE TRANSCRIPT EVERY TIME. Evidence arrives in
pieces: a recording, then a document, then another recording, minutes apart.
Extraction re-runs across the FULL accumulation every time, never
incrementally over the new chunk. A later statement can correct an earlier
one ("no, it's the right calf, not the left") and only a model seeing both
can resolve that; merging per-chunk findings would keep both as true. This
costs a full call per update, and the disk cache in llm.py absorbs the
repeats.

TWO THINGS SURVIVE ACROSS UPDATES, and the client sends them back on every
call: confirmed_exclusions (once a clinician ticks "PE ruled out by CTPA",
that holds for the rest of the consult) and asked (findings already put to
the patient, so the panel never re-asks). Candidates, ordering and the next
question all recompute fresh.

THERE IS NO OFFLINE REPLAY MODE OR PRE-COMPUTED PAYLOAD. Earlier versions
replayed canned results during supposedly live demos, showing output that
had nothing to do with the person in the room.
Everything on screen comes from this patient, this session, or the panel
says plainly that it has nothing.

Usage:
    pip install -r requirements.txt
    python main.py            # http://127.0.0.1:8000/web/
    python main.py --port 8080
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

import capture
import extract
import propose
import vocabulary
from engine import Engine

REPO_ROOT = Path(__file__).parent

app = FastAPI(title="AnviksaAI")

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = Engine.load()
    return _engine


def _log_unrated(rows):
    """Record rulebook gaps without any consultation evidence or identifiers."""
    if not rows:
        return
    try:
        path = REPO_ROOT / "cache" / "unrated_log.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as log:
            for row in rows:
                log.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "condition": row["condition"],
                }, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Losing an operational log must not suppress the clinical panel.
        print(f"unrated log write failed: {exc}", file=sys.stderr)


class AnalyzeRequest(BaseModel):
    transcript: str = ""
    doc_text: str = ""
    # Conditions a CLINICIAN has confirmed as ruled out. Never inferred here.
    confirmed_exclusions: list[str] = []
    # Findings already put to the patient but not yet answered.
    asked: list[str] = []


@app.post("/capture")
async def capture_only(request: Request):
    """STT or OCR only — no extract(), no propose(), no engine.

    Exactly one "audio" file, or one-or-more "documents", per call. That
    mirrors how liveSend()/docSend() in web/index.html call it, so there is
    no need to support a mixed audio+documents request.
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(400, "/capture requires multipart/form-data")

    form = await request.form()
    tmp_paths = []
    try:
        upload = form.get("audio")
        if upload is not None:
            suffix = Path(getattr(upload, "filename", "") or "").suffix or ".webm"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await upload.read())
                audio_tmp_path = Path(tmp.name)
            tmp_paths.append(audio_tmp_path)
            language_code = form.get("language_code") or "hi-IN"
            try:
                stt_result = await run_in_threadpool(
                    capture.transcribe, str(audio_tmp_path), language_code=language_code)
            except Exception as e:
                # A caught exception turned into an HTTPException prints no
                # traceback, which left a real live 502 with nothing in the
                # access log to diagnose it from. Printed to server stderr.
                traceback.print_exc()
                raise HTTPException(502, f"transcription failed: {e}")
            transcript = stt_result["transcript"]
            result = {"transcript": transcript}
            # English gloss, so a clinician who does not read the selected
            # language can still check what was just recorded. Best-effort:
            # a translation failure omits the field rather than failing the
            # capture that succeeded.
            if transcript.strip() and language_code != "en-IN":
                try:
                    result["transcript_en"] = await run_in_threadpool(
                        capture.translate, transcript,
                        source_language_code=language_code, target_language_code="en-IN")
                except Exception:
                    pass
            return result

        doc_uploads = form.getlist("documents")
        if doc_uploads:
            if any(not isinstance(upload, UploadFile) for upload in doc_uploads):
                raise HTTPException(502, "Malformed document request: documents must be file uploads")
            docs_meta = []
            for doc_upload in doc_uploads:
                name = getattr(doc_upload, "filename", "") or "document"
                suffix = Path(name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(await doc_upload.read())
                    doc_tmp_path = Path(tmp.name)
                tmp_paths.append(doc_tmp_path)
                docs_meta.append((name, doc_tmp_path))

            # OCR every document CONCURRENTLY. These are independent vision
            # calls, so a doctor uploading three records at once was paying
            # three times the wall clock for no reason. The 10-30 req/min
            # vision cap is a per-minute ceiling; what that warning is
            # actually about is re-scanning inside a render loop, not one
            # batched upload.
            async def _scan(name, path):
                try:
                    result = await run_in_threadpool(capture.read_image, str(path))
                    text = result["text"]
                    # Screened per document, HERE, because this is the last point
                    # at which per-document text exists. /analyze concatenates
                    # everything into one blob, after which a game screenshot's
                    # words are indistinguishable from a prescription's — which
                    # is exactly how one changed the candidate list.
                    screen = await run_in_threadpool(extract.screen_document, text)
                    return {"name": name, "chars": len(text), "text": text,
                            "clinical": screen["clinical"], "kind": screen["kind"]}
                except Exception as exc:
                    traceback.print_exc()
                    return {"name": name, "status": "error", "clinical": False,
                            "chars": 0, "text": "", "kind": "could not be read",
                            "error": f"Could not read document ({type(exc).__name__}). Try scanning it again."}

            scans = asyncio.gather(*[_scan(n, p) for n, p in docs_meta])
            try:
                results = await asyncio.shield(scans)
            finally:
                # Even a cancelled request must wait for its worker threads
                # before the outer finally deletes their input files.
                await scans

            return {"scanned_documents": results}

        raise HTTPException(400, "/capture requires an 'audio' file or 'documents' files")
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


def _pipeline(req: AnalyzeRequest):
    """extract -> propose -> triage. Two model calls, never one.

    Synchronous and blocking (two chat completions, tens of seconds), which
    is why the route below runs it in a threadpool rather than calling it
    directly: called inline from an async handler it froze uvicorn's event
    loop for the whole duration and refused even new incoming connections.
    Confirmed live — the third of three sequential requests got
    ERR_CONNECTION_REFUSED while the second was still blocking.
    """
    findings, quotes, other = extract.extract(req.transcript, req.doc_text, include_other=True)
    candidates = propose.propose(findings, other)
    payload = get_engine().assess(
        candidates, findings,
        confirmed_exclusions=req.confirmed_exclusions,
        asked=[fid for fid in req.asked if fid in vocabulary.FINDINGS],
        other_findings=other,
    )
    _log_unrated(payload["unrated"])
    # The quote behind each finding. With no dataset and no eval, a clinician
    # being able to trace a finding back to the words that produced it is a
    # real part of the safety net — and it is how a fabricated finding gets
    # caught.
    payload["_findings"] = findings
    payload["_quotes"] = quotes
    payload["other_findings"] = other
    return payload


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        return await run_in_threadpool(_pipeline, req)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(502, f"analysis failed: {e}")


class SpeakRequest(BaseModel):
    # The finding id whose question to ask. NOT free text: the server looks
    # the wording up itself, so nothing the browser sends can put words in
    # the clinician's mouth or bypass the clinician-only check below.
    finding: str
    language_code: str = "hi-IN"


@app.post("/speak")
async def speak(req: SpeakRequest):
    """Ask the current question aloud, in the patient's language.

    -> {"text": <what is actually spoken>, "audio_b64": <wav>, "translated": bool}

    THE TRANSLATION IS RETURNED SO IT CAN BE SHOWN. Sarvam's translation
    quality on clinical phrasing across ten Indian languages has not been
    verified by a native speaker, and a mistranslated question asked aloud
    produces a wrong answer that enters the findings pipeline as genuine
    evidence — indistinguishable, afterwards, from something the patient
    actually said. Putting the translated sentence on screen next to the
    audio does not fix that, but it makes it inspectable instead of
    invisible. Anyone who reads the language can catch it.

    The previous build had none of this: it shipped one hardcoded Hindi
    sentence for one question id and spoke raw English in an English voice
    for everything else. It went unnoticed because the scripted demo pinned
    the question to that one id.
    """
    ask = vocabulary.question(req.finding)
    if not ask:
        raise HTTPException(400, f"no such finding: {req.finding}")

    # Examination and observation items are written for the clinician.
    # "Are the neck veins distended?" read aloud to a patient is nonsense,
    # and refusing here rather than in the frontend means no caller can get
    # it wrong.
    if vocabulary.is_clinician_only(req.finding):
        raise HTTPException(400, "this question is for the clinician and is not spoken to the patient")

    text, translated = ask, False
    if req.language_code != "en-IN":
        try:
            text = await run_in_threadpool(
                capture.translate, ask,
                source_language_code="en-IN", target_language_code=req.language_code)
            translated = True
        except Exception:
            # Speaking English in an English voice would be the old bug.
            # Better to fail the button than to ask the patient a question
            # in a language they may not have.
            traceback.print_exc()
            raise HTTPException(502, "could not translate the question; not speaking it")

    try:
        audio = await run_in_threadpool(capture.speak, text, req.language_code)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(502, f"speech synthesis failed: {e}")

    return {
        "text": text,
        "translated": translated,
        "audio_b64": base64.b64encode(audio).decode("ascii"),
    }


@app.get("/")
def root():
    return RedirectResponse("/web/")


# Mounted at /web, NOT at "/". Mounting the repo root would also serve .env
# and .git/ over HTTP.
app.mount("/web", StaticFiles(directory=str(REPO_ROOT / "web"), html=True), name="web")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    # PORT is the env var hosting platforms inject and expect the app to bind
    # to. --host defaults to 0.0.0.0 because a loopback-only bind is
    # unreachable from outside a container; harmless locally, since localhost
    # still resolves to it.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    import uvicorn
    print(f"Serving on http://{args.host}:{args.port}/web/")
    uvicorn.run(app, host=args.host, port=args.port)
