"""llm.py — the one place that talks to Sarvam's chat model.

extract.py and propose.py both need the same four things: a client with a
sane timeout, retry on the transient failures every provider this project
has used exhibits, a disk cache keyed by content hash, and forced tool
calling for structured output. That plumbing lived in extract.py and was
about to be copy-pasted into propose.py, so it moved here instead.

WHAT THIS FILE DOES NOT DO: it does not know what a finding is, what a
condition is, or that this is a medical product at all. It takes a system
prompt, a user message and a tool schema, and returns parsed arguments.
Keeping it ignorant is deliberate — the two callers must stay two genuinely
separate calls with separate prompts, and a shared helper that started
assembling clinical prompts would be the first step back toward one call.

WHY TWO CALLS, NEVER ONE (the rule this file is built to preserve):
extraction must not see the candidate list. Give a model the transcript and
a list of conditions in the same breath and it starts extracting findings
*because* they fit a condition it is already entertaining. That is not
hypothetical — an earlier provider fabricated an immobility finding on a
transcript that never mentioned immobility, and this project switched
providers over exactly that failure.

DETERMINISM. temperature=0 and a fixed seed, always. Confirmed live on this
model: with no sampling params set, three back-to-back calls on identical
input returned different results, which changed what was on screen for
reasons that had nothing to do with the patient. The seed moves only on
retry — see call_tool().

PROVIDER HISTORY (see git log): Gemini fabricated present findings for
topics never discussed; Groq's org cap forced the catalog into four batches;
Sarvam fits the whole thing in one request and is already the provider
capture.py uses for STT, vision and TTS, so it needs no second credential.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import httpx
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

MODEL = os.environ.get("SARVAM_CHAT_MODEL", "sarvam-105b")
# Not sarvam-105b-conversations — that variant is tuned for real-time voice
# turns, not one-shot structured extraction.

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def content_hash(*parts):
    """Short stable hash of the real inputs to a call. Same convention as
    capture.py's STT/OCR cache."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:24]


_client = None


def client():
    global _client
    if _client is None:
        # timeout=60.0: real successful calls on this model have been
        # measured at 32-50s, so 60 leaves headroom without letting a dead
        # call hang for the length of a whole consultation.
        _client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"], timeout=60.0)
    return _client


class ToolCallMissing(Exception):
    """A forced-tool-call response came back with no tool_calls anyway.

    Observed live on sarvam-105b, and previously on Groq's gpt-oss-20b as
    "tool_use_failed". Confirmed transient both times by retrying the same
    input and having it succeed, so it is treated as retryable rather than
    as a hard failure on first occurrence."""


def call_tool(system_prompt, user_content, tool, *, max_tokens=4096, max_retries=2):
    """One forced-tool-call round trip. Returns the parsed arguments dict.

    Retries on 429/5xx, on network and timeout errors, and on ToolCallMissing.

    THE TIMEOUT BUDGET HAS TO BE READ END TO END, not per call. /analyze
    makes TWO of these, so the worst case here doubles before the browser
    sees anything. max_retries=2 at a 60s client timeout is 122s per call and
    ~244s for the pair; at 3 retries it was 186s and ~372s.

    That number has to stay BELOW the frontend's abort timer, or the browser
    kills work that is still running and the clinician sees a stale panel
    with no error. It was above it: the frontend aborted at 75s while a
    single cold extraction measured 120s — one Sarvam timeout plus a
    successful retry. The abort exists to recover a genuinely dead
    connection, not to cap slow work, so it belongs above this ceiling AND
    above the cold start that can precede it on a free host that sleeps
    after 15 idle minutes. It sits at 360s: 244 + 60, with margin.

    Anything hosting this needs to allow a request that long too; most
    proxies cap well below it by default. Vercel and Netlify functions cap
    at 10-60s and cannot host this at all. See render.yaml.
    """
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client().chat.completions(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": tool["function"]["name"]}},
                temperature=0.0,
                # The seed moves ONLY on retry. Observed live: with
                # temperature=0 and a fixed seed, a ToolCallMissing response
                # is perfectly deterministic — every retry sent the identical
                # request and got the identical empty response back, five
                # times, a guaranteed failure. Attempt 0 keeps seed=0, so a
                # normal successful call stays reproducible.
                seed=attempt,
                max_tokens=max_tokens,
            )
            tool_calls = getattr(resp.choices[0].message, "tool_calls", None)
            if not tool_calls:
                raise ToolCallMissing("response had no tool_calls despite forced tool_choice")
            # .arguments is a JSON STRING on this SDK (confirmed from the
            # FunctionCall type's `arguments: str` field and a real call),
            # unlike Gemini's already-parsed dict.
            return json.loads(tool_calls[0].function.arguments)

        # httpx errors arrive UNWRAPPED from the sarvamai SDK's underlying
        # client — they are not ApiError, so an except clause listing only
        # ApiError never caught them and a transient network hiccup failed
        # hard on the first attempt with no retry at all. Observed live,
        # twice in a row, mid-demo.
        except (ApiError, ToolCallMissing, httpx.TimeoutException, httpx.NetworkError) as e:
            status = getattr(e, "status_code", None) if isinstance(e, ApiError) else None
            retryable = (
                isinstance(e, (ToolCallMissing, httpx.TimeoutException, httpx.NetworkError))
                or status in (429, 500, 502, 503, 504)
            )
            if retryable and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


def cached(path, produce):
    """Read `path` if it exists, else call produce() and write the result.

    Every model call in this project is cached on generation. Sarvam's vision
    endpoint is capped at 10-30 requests/minute on every tier, and re-running
    an identical transcript (a UI retry, a test iterating on the same case)
    should never burn a fresh call.
    """
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = produce()
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result
