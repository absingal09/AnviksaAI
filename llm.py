"""llm.py — task-specific chat models behind one Sarvam credential.

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
Current defaults: Gemma 4 screens documents; GLM 5.3 extracts and proposes
in separate requests. These models use V2; sarvam-105b remains available on V1.
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

_DEFAULT_MODELS = {"screen": "gemma4", "extract": "glm5.3", "propose": "glm5.3"}
_V2_MODELS = {"glm5.3", "gemma4", "deepseekv4.1-flash"}
_V1_MODELS = {"sarvam-105b", "sarvam-105b-conversations"}


def config(task):
    """Independent model profiles; the legacy setting still overrides all defaults."""
    if task not in _DEFAULT_MODELS:
        raise ValueError(f"Unknown model task: {task}")
    model = (os.environ.get(f"SARVAM_{task.upper()}_MODEL")
             or os.environ.get("SARVAM_CHAT_MODEL") or _DEFAULT_MODELS[task]).strip()
    if model not in _V1_MODELS | _V2_MODELS:
        raise ValueError(f"Unsupported Sarvam model for {task}: {model}")
    reasoning = model in {"glm5.3", "deepseekv4.1-flash"}
    return {
        "model": model,
        "api_version": "v2" if model in _V2_MODELS else "v1",
        "reasoning_effort": ("low" if task == "screen" else "high") if reasoning else None,
        # Reasoning and tool arguments share the output budget. Never give a
        # reasoning model the tiny budget used for direct classification.
        "max_tokens": (4096 if task == "screen" else 16384) if reasoning
                      else (256 if task == "screen" else 8192),
    }


def config_key(task):
    """Everything about model routing/sampling that affects a cached response."""
    return json.dumps({**config(task), "temperature": 0.0, "seed": 0}, sort_keys=True)

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
_http_client = None


def client():
    global _client
    if _client is None:
        # timeout=60.0: real successful calls on this model have been
        # measured at 32-50s, so 60 leaves headroom without letting a dead
        # call hang for the length of a whole consultation.
        _client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"], timeout=60.0)
    return _client


def http_client():
    """The installed SDK predates V2; use its documented HTTP API directly."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=60.0)
    return _http_client


class ToolCallMissing(Exception):
    """A forced-tool-call response came back with no tool_calls anyway.

    Observed live on sarvam-105b, and previously on Groq's gpt-oss-20b as
    "tool_use_failed". Confirmed transient both times by retrying the same
    input and having it succeed, so it is treated as retryable rather than
    as a hard failure on first occurrence."""


def call_tool(system_prompt, user_content, tool, *, task="extract", max_tokens=None, max_retries=2):
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
    settings = config(task)
    delay = 2.0
    for attempt in range(max_retries):
        try:
            parameters = dict(
                model=settings["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": tool["function"]["name"]}},
                # Explicit null disables Sarvam's default thinking mode for
                # direct models; reasoning models get a separate larger budget.
                reasoning_effort=settings["reasoning_effort"],
                temperature=0.0,
                # The seed moves ONLY on retry. Observed live: with
                # temperature=0 and a fixed seed, a ToolCallMissing response
                # is perfectly deterministic — every retry sent the identical
                # request and got the identical empty response back, five
                # times, a guaranteed failure. Attempt 0 keeps seed=0, so a
                # normal successful call stays reproducible.
                seed=attempt,
                max_tokens=max_tokens if max_tokens is not None else settings["max_tokens"],
            )
            if settings["api_version"] == "v2":
                response = http_client().post(
                    "https://api.sarvam.ai/v2/chat/completions", json=parameters,
                    headers={"api-subscription-key": os.environ["SARVAM_API_KEY"]},
                )
                response.raise_for_status()
                parsed = response.json()
            else:
                parsed = client().chat.completions(**parameters).model_dump()
            choices = parsed.get("choices") if isinstance(parsed, dict) else None
            first = choices[0] if isinstance(choices, list) and choices else None
            message = first.get("message") if isinstance(first, dict) else None
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            if not isinstance(tool_calls, list) or not tool_calls:
                raise ToolCallMissing("response had no tool_calls despite forced tool_choice")
            # .arguments is a JSON STRING on this SDK (confirmed from the
            # FunctionCall type's `arguments: str` field and a real call),
            # unlike Gemini's already-parsed dict.
            expected_name = tool["function"]["name"]
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                if isinstance(function, dict) and function.get("name") == expected_name:
                    raw_arguments = function.get("arguments")
                    if not isinstance(raw_arguments, str):
                        continue
                    arguments = json.loads(raw_arguments)
                    if isinstance(arguments, dict):
                        return arguments
            raise ToolCallMissing("response did not return the requested structured object")

        # httpx errors arrive UNWRAPPED from the sarvamai SDK's underlying
        # client — they are not ApiError, so an except clause listing only
        # ApiError never caught them and a transient network hiccup failed
        # hard on the first attempt with no retry at all. Observed live,
        # twice in a row, mid-demo.
        except (ApiError, ToolCallMissing, json.JSONDecodeError, httpx.HTTPStatusError,
                httpx.TimeoutException, httpx.NetworkError) as e:
            status = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else getattr(e, "status_code", None)
            retryable = (
                isinstance(e, (ToolCallMissing, json.JSONDecodeError, httpx.TimeoutException, httpx.NetworkError))
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
