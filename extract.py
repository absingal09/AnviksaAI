"""extract.py — the translator.

Turns this patient's consultation transcript, plus whatever documents they
brought, into finding ids from findings.yaml. Three-valued:

    {id: True}   finding confirmed PRESENT
    {id: False}  finding EXPLICITLY DENIED
    id omitted   nobody discussed it — UNKNOWN

THE ONE HARD RULE: never guess False for something nobody asked about. An
unasked symptom is not a denied symptom. Marking undiscussed findings absent
systematically penalises exactly the conditions Can't-Miss exists to surface
— the panel would look confident while quietly suppressing the diagnoses it
is there to catch. This file's entire job is preserving that three-way
distinction.

EXTRACTION NEVER SEES THE CANDIDATE LIST. This module receives a transcript
and a vocabulary, and nothing else. propose.py, which does the reasoning,
receives findings and never sees the transcript. Collapsing the two into one
call makes the model extract findings *because* they fit a condition it is
already considering — an earlier provider fabricated an immobility finding on
a transcript that never mentioned immobility, and this project switched
providers over that failure. The separation is the fix; do not merge them.

RE-EXTRACTION IS OVER THE WHOLE THING, ALWAYS. A consult is not one input —
it is a recording, then a document, then another recording, landing minutes
apart. Every update re-runs this over the FULL accumulated transcript and
document text, never incrementally over just the new chunk. A later statement
can correct an earlier one ("no, it's the right calf, not the left"), and only
a model seeing both can resolve that; merging per-chunk findings would keep
both as true. This costs a full call per update. Pay it.

Returned ids are validated against the vocabulary and unknown ones dropped.
The model is instructed to use only ids from the catalog, but "instructed to"
is not "guaranteed to", and a fabricated id would flow downstream as a
finding no clinician could trace to anything the patient said.
"""

import json
import re

import llm
import vocabulary
from propose import _UNSUPPORTED_REASONING

SYSTEM_PROMPT = """You are a clinical scribe. You read a consultation \
transcript and any documents the patient brought, and you record which \
findings from a fixed catalog were actually discussed.

You are NOT diagnosing. You are NOT deciding what matters. Another system \
does that, and it never sees this transcript. Your only job is to report \
faithfully what was said.

THREE-VALUED. Every finding is in exactly one of three states:
  PRESENT  — the patient described it, or a document shows it.
  ABSENT   — the patient was asked and denied it, or explicitly volunteered \
that they do not have it.
  UNKNOWN  — anything else. This is the default and it covers most of the \
catalog on most consultations.

Report present findings and absent findings. Report NOTHING for unknown ones.

THE MISTAKE THAT MATTERS MOST: marking something ABSENT because it was not \
mentioned. If nobody in the transcript raised the topic of leg swelling, leg \
swelling is UNKNOWN, not absent. "Not mentioned" and "denied" are completely \
different pieces of information and confusing them is the single worst error \
you can make here. When in doubt, leave it out.

THE SECOND MISTAKE: recording a finding because it fits a pattern you \
recognise. If a patient describes sudden breathlessness you may find \
yourself reaching for the rest of a picture — pleuritic pain, a swollen \
calf, a recent flight. Record only what is in the text. You do not know what \
condition anyone is considering, and that is deliberate.

RULES
- For present and absent, use ONLY ids that appear in the catalog. Never invent an id.
- Use a catalog id whenever one fits. Use `other` ONLY for clinically relevant \
present statements that no catalog finding covers. Never put a denial, an \
absence, an unknown, or a hypothetical in `other`. Return at most ten entries, \
each a short clinical phrase in English of at most twelve words, with a quote \
copied from the transcript or documents. Do not write percentages, probability, \
likelihood, confidence, or "chance of" claims in the phrase. Keep the evidence \
quote verbatim, including any numbers or percentages the source contains.
- Documents count as evidence. A photographed prescription or blister strip \
naming a drug establishes that the patient takes that drug, even if they \
never said so out loud. Read the drug name and map it to the right finding \
— a strip of desogestrel with ethinylestradiol is a combined oral \
contraceptive.
- Things said in passing count. A long journey mentioned while answering a \
question about something else is still a long journey.
- Third parties do not count. "My sister gets anxiety" is not a finding \
about this patient.
- Hypotheticals and reassurance do not count. A doctor saying "it might be \
acidity" is not the patient reporting heartburn.
- A finding marked (Examination) or (Observation) is recorded only if the \
clinician actually states that finding during the consultation. Do not infer \
examination findings from symptoms.
- If two findings are opposites and the patient described one, you may mark \
the other absent — sudden onset described means gradual onset absent. Only \
do this when the transcript genuinely settles it.
- If neither the transcript nor the supplied documents contains clinical content, \
record nothing. An empty result is a correct and expected answer.

For engineering test inputs, sample or synthetic labels do not erase the clinical facts \
explicitly written in the supplied records. Extract those facts for the test patient \
using the same evidence requirements. A blank transcript does not mean the DOCUMENTS \
section is blank. Do not infer additional symptoms or diagnoses.
"""

_TOOL = {
    "type": "function",
    "function": {
        "name": "record_findings",
        "description": (
            "Record the findings actually discussed in this consultation. "
            "Omit anything nobody raised — omission means UNKNOWN, which is "
            "different from absent and is the correct answer for most of "
            "the catalog on most consultations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "present": {
                    "type": "array",
                    "description": "Findings the patient described, or a document shows.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "A finding id, exactly as it appears in the catalog.",
                            },
                            "evidence": {
                                "type": "string",
                                "description": (
                                    "The words from the transcript or document "
                                    "that establish this. Quote them; do not "
                                    "paraphrase. If you cannot quote anything, "
                                    "the finding does not belong here."
                                ),
                            },
                        },
                        "required": ["id", "evidence"],
                    },
                },
                "absent": {
                    "type": "array",
                    "description": (
                        "Findings the patient was asked about and denied, or "
                        "explicitly volunteered that they do not have. NOT "
                        "findings that simply went unmentioned."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "evidence": {
                                "type": "string",
                                "description": "The words in which it was denied. Quote them.",
                            },
                        },
                        "required": ["id", "evidence"],
                    },
                },
                "other": {
                    "type": "array",
                    "maxItems": 10,
                    "description": "Present clinical statements not covered by any catalog finding; never denials or absences.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Short clinical phrase in English, at most twelve words. Use a catalog id instead whenever one fits."},
                            "evidence": {"type": "string", "description": "Verbatim quote copied from the transcript or documents."},
                        },
                        "required": ["text", "evidence"],
                    },
                },
            },
            "required": ["present", "absent", "other"],
        },
    },
}


# Everything except the patient's own words that can change the answer:
# the system prompt, the tool schema, and the vocabulary the model is given.
# Folded into the cache key so an edit to any of them invalidates cached
# results instead of being silently ignored.
_PROMPT_VERSION = llm.content_hash(
    SYSTEM_PROMPT, json.dumps(_TOOL, sort_keys=True),
    # The WHOLE catalog, not just the ids. The catalog sent to the model
    # carries each finding's label and question, so editing a question
    # changes the prompt. Hashing ids alone would have served the old
    # extraction back — the same stale-cache bug this line was added to fix,
    # one level down.
    json.dumps(vocabulary.catalog_for_prompt(), sort_keys=True, ensure_ascii=False),
)


_SCREEN_PROMPT = """You are checking whether a photographed document belongs \
in a patient's clinical record.

You are given the text an OCR pass pulled out of one image. Decide whether \
that image is a CLINICAL document — something a patient would hand a doctor, \
or a doctor would file.

CLINICAL: prescriptions, medicine strips and packaging, lab and imaging \
reports, discharge summaries, referral letters, vaccination cards, hospital \
cards, previous consultation notes, ECG printouts, medical bills that name \
drugs or investigations.

NOT CLINICAL: screenshots of games, apps, websites and chats; photographs of \
people, places, food or pets; receipts and bills for anything other than \
care; identity documents with no medical content; blank or unreadable pages; \
anything where the OCR text is noise.

OCR is imperfect. Judge by what the text is plainly ABOUT, not by whether it \
is clean. A smudged prescription is still a prescription. If a document is \
genuinely ambiguous, call it clinical — a clinician can dismiss one extra \
document in a second, and wrongly discarding a real prescription throws away \
evidence nobody will ever know was missing.

`kind` is a two-to-four word description a clinician can read at a glance: \
"oral contraceptive strip", "chest X-ray report", "mobile game screenshot"."""

_SCREEN_TOOL = {
    "type": "function",
    "function": {
        "name": "screen_document",
        "description": "Say whether this image is a clinical document.",
        "parameters": {
            "type": "object",
            "properties": {
                "clinical": {
                    "type": "boolean",
                    "description": "True if this belongs in a clinical record.",
                },
                "kind": {
                    "type": "string",
                    "description": "Two to four words describing what it is.",
                },
            },
            "required": ["clinical", "kind"],
        },
    },
}


def screen_document(text):
    """One document's OCR text -> {"clinical": bool, "kind": str}.

    WHY THIS EXISTS. Document text is concatenated into one blob before
    extraction, so a photograph of anything at all contributes its words to
    the same input a prescription does. Observed live: uploading a game
    screenshot alongside a real consultation changed the candidate list, and
    added a finding that nothing the patient said supported. The screen runs
    per document, while per-document text still exists — after concatenation
    there is no way to attribute anything to anything.

    THIS FLAGS; IT DOES NOT DELETE. The caller shows the judgment and lets a
    clinician override it. Wrongly discarding a real prescription throws away
    evidence nobody will ever know was missing, which is the same failure
    shape as removing a warning, so a human keeps the last word here too.

    Deliberately NOT part of extract(): this judges what a document IS, and
    it must not see the transcript or the vocabulary. A screen that knew what
    findings were being looked for would start rating documents by whether
    they were useful rather than by whether they were real.
    """
    text = (text or "").strip()
    if not text:
        return {"clinical": False, "kind": "no readable text"}

    cpath = llm.CACHE_DIR / f"screen_{llm.content_hash(llm.config_key('screen'), _SCREEN_PROMPT, json.dumps(_SCREEN_TOOL, sort_keys=True), text)}.json"

    def produce():
        return llm.call_tool(
            _SCREEN_PROMPT,
            f"OCR TEXT FROM ONE IMAGE:\n{text}\n\nCall screen_document.",
            _SCREEN_TOOL,
            task="screen",
        )

    try:
        parsed = llm.cached(cpath, produce)
    except Exception as e:
        # A screening failure must not cost the clinician a real document.
        # Fail open, and say so rather than silently passing it through.
        print(f"screen_document: failed, treating as clinical — {e}")
        return {"clinical": True, "kind": "not screened"}

    if not isinstance(parsed, dict):
        return {"clinical": True, "kind": "not screened"}
    clinical = parsed.get("clinical")
    if isinstance(clinical, str) and clinical.strip().lower() in ("true", "false"):
        clinical = clinical.strip().lower() == "true"
    if not isinstance(clinical, bool):
        return {"clinical": True, "kind": "not screened"}
    kind = parsed.get("kind")
    return {
        "clinical": clinical,
        "kind": kind.strip() if isinstance(kind, str) and kind.strip() else "unrecognised",
    }


def _flatten(transcript):
    """Accepts a plain string, or a list of utterance dicts as returned by
    capture.transcribe()'s diarized entries."""
    if isinstance(transcript, (list, tuple)):
        lines = []
        for u in transcript:
            speaker = u.get("speaker") or u.get("w") or "?"
            text = u.get("text") or u.get("transcript") or u.get("e") or u.get("orig") or ""
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)
    return transcript or ""


def _normalise_quote(text):
    return " ".join(text.casefold().split())


_NEGATED_QUOTE = re.compile(
    r"\b(?:no|denies|denied|deny|not|without|never|cannot|negative)\b|\b\w+n['’]t\b",
    re.IGNORECASE,
)


def _phrase_key(text):
    """Case/punctuation/whitespace normalisation, not semantic equivalence."""
    return tuple(re.findall(r"[^\W_]+", text.casefold()))


def _catalog_overlap(text, evidence, findings, quotes):
    """English token-containment heuristic, NOT a clinical synonym resolver.

    Only already-recorded catalog findings participate. The identical-quote
    check accepts only grammar/dose words beyond the supplemental phrase,
    so another drug in a shared quote does not become a duplicate by accident.
    Combined phrases can still overmatch; synonyms can escape. Do not
    turn this heuristic into new catalog findings or change their states.
    """
    stop = {"a", "an", "the", "i", "my", "on", "in", "of", "have", "has", "take", "taking"}
    words = set(_phrase_key(text)) - stop
    if not words:
        return None
    for fid in findings:
        for form in (vocabulary.label(fid), fid.replace("_", " ")):
            covered = set(_phrase_key(form)) - stop
            if covered and (covered <= words or words <= covered):
                return fid
        quote = quotes.get(fid, "")
        if quote and _normalise_quote(evidence) == _normalise_quote(quote):
            quoted_words = set(_phrase_key(quote)) - stop
            dose_words = {"mg", "mcg", "g", "ml", "daily", "tablet", "tablets", "once", "twice"}
            remainder = quoted_words - words - dose_words
            if words <= quoted_words and all(word.isdigit() for word in remainder):
                return fid
    return None


def _other_findings(items, sources, findings=(), quotes=None):
    """Validate a separate positive-evidence channel; never create catalog IDs.

    The shared proposal regex checks model-written text only. Source quotes
    retain doses, durations and percentages exactly as supplied.
    """
    if not isinstance(items, list):
        return []
    other, seen = [], set()
    for item in items:
        text = item.get("text") if isinstance(item, dict) else None
        evidence = item.get("evidence") if isinstance(item, dict) else None
        text = " ".join(text.split()) if isinstance(text, str) else ""
        key = _phrase_key(text)
        reason = None
        if not text or len(text.split()) > 12:
            reason = "missing text or more than twelve words"
        elif _UNSUPPORTED_REASONING.search(text):
            reason = "probability wording in generated text"
        elif not isinstance(evidence, str) or not evidence.strip() or not any(
                _normalise_quote(evidence) in source for source in sources):
            reason = "evidence quote not found verbatim in source"
        # Conservative English negation heuristic: whole words/contractions,
        # not substrings (nosebleeds stays positive). Mixed clauses and
        # "not only" can be false positives; other languages can be missed.
        # "cannot stop" / "can't stop" instead assert an ongoing symptom.
        elif _NEGATED_QUOTE.search(re.sub(r"\b(?:cannot|can['’]t)\s+stop\b", "", evidence,
                                         flags=re.IGNORECASE)):
            reason = "negated source quote (English heuristic)"
        elif (covered := _catalog_overlap(text, evidence, findings, quotes or {})):
            reason = f"overlaps catalog finding {covered} (token heuristic)"
        elif key in seen:
            reason = "duplicate supplemental phrase"
        elif len(other) >= 10:
            reason = "more than ten entries"
        if reason:
            print(f"extract: dropped other {text or '<malformed>'!r}: {reason}")
            continue
        other.append({"id": f"other:{len(other)}", "text": text, "evidence": evidence.strip()})
        seen.add(key)
    return other


def extract(transcript, doc_text="", *, include_other=False):
    """-> ({finding_id: True|False}, {finding_id: "quoted evidence"})

    The second return value is the quote that justifies each finding. It is
    not decoration: with no dataset and no eval, a clinician being able to
    trace a finding back to the words that produced it is a real part of the
    safety net. It is also how a fabricated finding gets caught — a model
    that cannot quote anything for a finding it recorded has invented it.

    With include_other=True, return (findings, quotes, other_findings).
    The default pair remains compatible with existing catalog-only callers.
    Supplemental IDs are ephemeral and must never enter catalog state.
    """
    transcript = _flatten(transcript)
    doc_text = doc_text or ""

    # Nothing to extract from. Skip the call entirely rather than asking a
    # model to find findings in an empty string — it costs money, takes
    # seconds, and invites invention.
    if not transcript.strip() and not doc_text.strip():
        return ({}, {}, []) if include_other else ({}, {})

    # The cache key covers EVERYTHING that determines the answer, not just the
    # patient's words. Keying on (model, transcript, doc_text) alone meant
    # adding findings to the vocabulary changed nothing for any transcript
    # already seen — the old, smaller extraction was served forever. Caught
    # live: three findings were added to fill a real gap, and re-running the
    # exact transcript that exposed the gap returned the same two findings as
    # before. A stale cache that silently drops findings is the same failure
    # shape as an extractor that misses them.
    cpath = llm.CACHE_DIR / f"extract_{llm.content_hash(llm.config_key('extract'), _PROMPT_VERSION, transcript, doc_text)}.json"

    def produce():
        catalog = vocabulary.catalog_for_prompt()
        user_content = (
            f"FINDING CATALOG ({len(catalog)} findings). Use only these ids:\n"
            f"{json.dumps(catalog, ensure_ascii=False)}\n\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            f"DOCUMENTS (prescriptions, blister strips, reports — may be empty):\n"
            f"{doc_text or '(none)'}\n\n"
            "Call record_findings. Include a finding only if you can quote the "
            "words that establish it. Leave everything else out."
        )
        return llm.call_tool(SYSTEM_PROMPT, user_content, _TOOL, task="extract")

    parsed = llm.cached(cpath, produce)

    if not isinstance(parsed, dict):
        return ({}, {}, []) if include_other else ({}, {})
    findings, quotes, dropped = {}, {}, []
    recorded_states = {}
    for state, key in ((True, "present"), (False, "absent")):
        items = parsed.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            fid = item["id"].strip()
            if fid not in vocabulary.FINDINGS:
                # Not in the vocabulary, so nothing downstream could match it
                # across turns even if it were real. Dropped, and reported —
                # a run that drops a lot of ids means the catalog is missing
                # something the model keeps reaching for.
                dropped.append(fid)
                continue
            recorded_states.setdefault(fid, set()).add(state)
            evidence = item.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                continue
            findings[fid] = state
            quotes[fid] = evidence.strip()

    # A contradictory model response does not settle the finding. In
    # particular, iteration order must never turn a present finding absent.
    for fid, states in recorded_states.items():
        if len(states) > 1:
            findings.pop(fid, None)
            quotes.pop(fid, None)

    if dropped:
        print(f"extract: dropped {len(dropped)} id(s) not in findings.yaml: {sorted(set(dropped))}")

    sources = [_normalise_quote(transcript), _normalise_quote(doc_text)]
    mismatches = sum(not any(_normalise_quote(quote) in source for source in sources)
                     for quote in quotes.values())
    print(f"extract: catalog evidence quote mismatches: {mismatches}")
    other = _other_findings(parsed.get("other"), sources, findings, quotes)
    return (findings, quotes, other) if include_other else (findings, quotes)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("usage: python extract.py <transcript-file> [document-file]")
        sys.exit(1)
    _t = Path(sys.argv[1]).read_text(encoding="utf-8")
    _d = Path(sys.argv[2]).read_text(encoding="utf-8") if len(sys.argv) > 2 else ""
    _f, _q = extract(_t, _d)
    for _fid, _state in sorted(_f.items(), key=lambda kv: (not kv[1], kv[0])):
        print(f"  {'present' if _state else 'absent ':>7}  {vocabulary.label(_fid):<38}  {_q.get(_fid, '')!r}")
    print(f"\n{len(_f)} finding(s); everything else is UNKNOWN.")
