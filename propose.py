"""propose.py — findings in, candidate conditions out.

This module receives a set of findings and NOTHING ELSE. It never sees the
transcript, never sees the documents, never sees the patient's words. That is
not an oversight to be tidied up later; it is the reason the extraction step
can be trusted. A model holding both the raw transcript and a list of
conditions starts extracting findings *because* they fit a condition it is
already entertaining, and this project has already been burned by exactly
that — an earlier provider fabricated an immobility finding on a transcript
that never mentioned immobility.

WHAT IT RETURNS, AND WHAT IT REFUSES TO RETURN.

It returns conditions worth not forgetting, each with the findings that
support it, the findings that argue against it, and a readable reason. It
does not return a probability, a score, a confidence, or a rank. Not because
the model cannot produce one — it will happily produce a number for anything
— but because that number would be unfalsifiable. There is no dataset behind
it, no calibration, and nothing downstream that could ever check it. Three
significant figures of fabricated precision, presented to a clinician under
time pressure, is worse than no number at all.

THE CANDIDATE LIST IS UNBOUNDED. The model is given no list of conditions to
choose from. The previous build scored a fixed set of 49 synthetic
pathologies that contained no aortic dissection, no subarachnoid haemorrhage,
no meningitis, no sepsis, no ectopic pregnancy, no torsion and no cauda
equina — while including Ebola and Chagas. A tool whose job is "what
catastrophic thing have you not excluded" cannot have a closed list of things
it is able to say.

severity.yaml is NOT that list. It rates conditions for how bad missing them
would be, and a condition proposed here that has no entry there still reaches
the panel, marked unrated. It is never silently dropped.

WHAT THIS MODULE HONESTLY IS. The candidates come from a general-purpose
language model. If a chat assistant would name aortic dissection on this
presentation, so will this. With no eval, this cannot claim to be more
accurate than a doctor typing the case into a chat window, and nobody should
say otherwise. What it can claim: it happens without anyone being asked to
type a prompt, it keeps state across a consult, and the dangerousness ranking
applied to its output is a reviewed file rather than a generation. The
reasoning below is the safety net, which is why every candidate must carry it.
"""

import json
import re

import llm
import vocabulary

_UNSUPPORTED_REASONING = re.compile(
    r"[%％]|\b(?:percent(?:age)?|per\s+cent|probabilit(?:y|ies)|likelihood|confidence)\b"
    r"|\bchance\s+of\b|\bp\s*(?:=|\()",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are helping a clinician in a two-minute outpatient \
consultation remember what is worth not forgetting.

You are given a set of findings about one patient. You are NOT given the \
transcript, and you should not ask for it. You are not the person who \
decides what happens to this patient.

WHAT YOU RETURN: conditions consistent with these findings, each with the \
findings that support it, the findings that argue against it, and a short \
reason a clinician can read and check in three seconds.

WHAT YOU MUST NOT RETURN:
- Any probability, percentage, score, confidence or likelihood. Not in the \
reasoning text either. There is no data behind such a number and it would be \
fabricated precision.
- A ranking. Order does not matter; another system orders these.
- A single answer. You are not diagnosing. Naming the diagnosis is exactly \
what this tool refuses to do.

WHAT TO INCLUDE
- The obvious, common, benign explanation, when one fits. A panel that only \
ever shows catastrophes gets ignored within a week, and the clinician's own \
working diagnosis appearing beside the dangerous one is what makes the list \
readable rather than contrarian.
- Any dangerous condition that fits, even partially, even if it is unlikely. \
This is the whole point. Unlikely-but-lethal belongs on the list; another \
system decides how it is displayed.
- Conditions that fit the findings TOGETHER rather than one at a time. Pain \
worse lying flat, better leaning forward, after a recent viral illness is a \
picture; each of those findings alone is nearly meaningless. Reason about \
combinations, not a checklist.

CAST WIDE EARLY, AND LET THE FINDINGS NARROW IT. Early in a consultation \
there are two or three vague complaints and almost nothing else. That is \
NOT a reason to stay silent — it is exactly when a clinician most needs the \
broad "do not forget about" list, because it is the moment before anything \
has been ruled out. A headache with a rash deserves to have meningitis said \
out loud even though almost every patient with a headache and a rash has \
neither.

So: propose whatever is genuinely consistent with what you were given, and \
say honestly how thin the support is by listing only the findings that \
really do support it. A condition resting on one vague finding will be shown \
as weakly supported, next to your reasoning, and the clinician can dismiss \
it in a second. That is the intended outcome — not silence.

Expect to name MORE conditions when the findings are thin and FEWER as they \
accumulate, which is the opposite of how it might feel. More findings mean \
more things contradicted and dropped; a short list is something you earn \
from evidence, not something to start with.

HOW MANY: three to ten. Do not pad to reach a number, and do not trim to \
look decisive.

RETURN AN EMPTY LIST ONLY IF THERE IS GENUINELY NOTHING CLINICAL HERE — no \
symptom, no sign, no medication, nothing a doctor would act on. Not merely \
because what you have is vague or could be a hundred things.

FOR EACH CANDIDATE
- supported_by: ids of findings this patient HAS that support it. Only ids \
listed as present below.
- opposed_by: ids of findings this patient was asked about and DENIED that \
argue against it. Only ids listed as absent below.
- unresolved: ids of findings NOT yet known for this patient whose answer \
would most change whether this condition belongs on the list. Only ids from \
the catalog that appear in neither the present nor the absent list. This is \
the most useful field you produce: it is what lets the clinician be asked \
one good question instead of twenty.
- reasoning: one or two sentences naming the actual findings. "Sudden \
breathlessness with chest tightness in a patient on a combined oral \
contraceptive after two days of bus travel" — not "clinical picture is \
suggestive". A clinician who cannot check your reasoning cannot catch your \
mistake, and there is nothing else here that would catch it."""

_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_candidates",
        "description": (
            "Record the conditions worth keeping in mind for this patient. "
            "Empty only if the findings contain no clinical content at all — "
            "not merely because they are vague."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "description": (
                        "Everything genuinely consistent with these findings, "
                        "widest early and narrowing as findings accumulate. "
                        "Empty only if there is no clinical content at all."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "condition": {
                                "type": "string",
                                "description": (
                                    "The condition, named as a clinician would "
                                    "write it. No abbreviations a colleague "
                                    "would have to guess at."
                                ),
                            },
                            "reasoning": {
                                "type": "string",
                                "description": (
                                    "One or two sentences naming the actual "
                                    "findings that put this on the list. No "
                                    "numbers, no percentages, no hedging "
                                    "phrases that say nothing."
                                ),
                            },
                            "supported_by": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Finding ids from the PRESENT list only.",
                            },
                            "opposed_by": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Finding ids from the ABSENT list only.",
                            },
                            "unresolved": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Finding ids that are currently UNKNOWN and "
                                    "whose answer would most change whether "
                                    "this condition belongs on the list."
                                ),
                            },
                        },
                        "required": ["condition", "reasoning", "supported_by", "unresolved"],
                    },
                }
            },
            "required": ["candidates"],
        },
    },
}


# Everything except the patient's own words that can change the answer:
# the system prompt, the tool schema, and the vocabulary the model is given.
# Folded into the cache key so an edit to any of them invalidates cached
# results instead of being silently ignored.
_PROMPT_VERSION = llm.content_hash(
    SYSTEM_PROMPT, json.dumps(_TOOL, sort_keys=True),
    json.dumps(sorted(vocabulary.FINDINGS), ensure_ascii=False),
)


# Label -> id, for resolving a model that answered with the human-readable
# name instead of the id. Lowercased so case never decides whether a
# clinician sees the evidence behind a candidate.
_BY_LABEL = {m["label"].lower(): fid for fid, m in vocabulary.FINDINGS.items()}


def _resolve(item):
    """A model's answer -> a finding id, or None.

    Accepts the id, the id in any case, or the panel label. Deliberately
    stops there: anything looser would start guessing which finding was
    meant, and a wrong guess here invents evidence rather than losing it.
    """
    if not isinstance(item, str):
        return None
    text = item.strip()
    if text in vocabulary.FINDINGS:
        return text
    lowered = text.lower()
    if lowered in vocabulary.FINDINGS:
        return lowered
    return _BY_LABEL.get(lowered)


def propose(findings):
    """findings: {finding_id: True|False} from extract.extract().

    -> [{"condition", "reasoning", "supported_by", "opposed_by", "unresolved"}]

    No probability, no score, no rank. engine.py applies severity and support
    ordering to this; nothing here is ordered meaningfully.
    """
    present = sorted(f for f, state in findings.items() if state is True)
    absent = sorted(f for f, state in findings.items() if state is False)

    # Nothing present means nothing to reason from. Denials alone cannot
    # support a candidate, and asking the model to produce a list anyway is
    # how a panel ends up showing generic prior-driven suggestions styled as
    # though they came from this patient. Abstain instead, visibly.
    if not present:
        return []

    cache_key = llm.content_hash(llm.MODEL, _PROMPT_VERSION, json.dumps(present), json.dumps(absent))
    cpath = llm.CACHE_DIR / f"propose_{cache_key}.json"

    def produce():
        def render(ids):
            return "\n".join(f"  {f}  ({vocabulary.label(f)})" for f in ids) or "  (none)"

        unknown = [f for f in vocabulary.FINDINGS if f not in findings]
        user_content = (
            f"PRESENT — this patient has these:\n{render(present)}\n\n"
            f"ABSENT — asked about and denied:\n{render(absent)}\n\n"
            f"UNKNOWN — not discussed, valid ids for `unresolved` "
            f"({len(unknown)} of them):\n{render(unknown)}\n\n"
            "Call propose_candidates."
        )
        return llm.call_tool(SYSTEM_PROMPT, user_content, _TOOL)

    parsed = llm.cached(cpath, produce)

    present_set, absent_set = set(present), set(absent)
    candidates = []
    for raw in parsed.get("candidates") or []:
        condition = (raw.get("condition") or "").strip()
        reasoning = (raw.get("reasoning") or "").strip()
        # Every candidate carries its reasoning. One that cannot justify
        # itself cannot be checked by the clinician, and the clinician is the
        # entire safety net — so it does not reach the panel at all.
        if not condition or not reasoning:
            continue
        if _UNSUPPORTED_REASONING.search(reasoning):
            # Prompts cannot enforce this product rule. Validate cached and
            # newly generated reasoning alike before it can reach a panel.
            print(f"propose: rejected unsupported probability wording for {condition!r}")
            continue

        def keep(key, allowed):
            """Resolve to real ids in the right state, then filter.

            The filter matters: a model claiming a finding supports a
            condition when the patient does not have that finding is
            asserting evidence that does not exist — the one thing a
            clinician skim-reading a panel would not catch.

            The RESOLVE step matters just as much, and was missing. Matching
            ids exactly meant a model answering "Headache" instead of
            "headache" had every supporting finding dropped, and the panel
            rendered a row of real candidates each captioned "no findings for
            it" while the same findings sat in Doesn't Fit saying nothing
            accounted for them. Observed live. Both halves of the screen were
            wrong, and neither looked like a bug.
            """
            out, dropped = [], []
            for item in raw.get(key) or []:
                fid = _resolve(item)
                if fid is not None and fid in allowed:
                    if fid not in out:
                        out.append(fid)
                elif item:
                    dropped.append(item)
            if dropped:
                # Printed, not swallowed. A candidate losing its evidence is
                # invisible on screen — it just looks like a weak candidate.
                print(f"propose: {condition!r} {key}: dropped {dropped}")
            return out

        candidates.append({
            "condition": condition,
            "reasoning": reasoning,
            "supported_by": keep("supported_by", present_set),
            "opposed_by": keep("opposed_by", absent_set),
            # Unresolved must be genuinely unknown: a real finding, and
            # neither present nor absent. Anything else produces a question
            # the patient has already answered, which is the bug this project
            # spent a whole build cycle on.
            "unresolved": keep("unresolved", set(vocabulary.FINDINGS) - set(findings)),
        })

    return candidates


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print('usage: python propose.py \'{"chest_tightness": true, "fever": false}\'')
        sys.exit(1)
    _f = json.loads(sys.argv[1])
    for _c in propose(_f):
        print(f"\n{_c['condition']}")
        print(f"  {_c['reasoning']}")
        print(f"  supported by: {', '.join(vocabulary.label(x) for x in _c['supported_by']) or '-'}")
        print(f"  opposed by:   {', '.join(vocabulary.label(x) for x in _c['opposed_by']) or '-'}")
        print(f"  unresolved:   {', '.join(vocabulary.label(x) for x in _c['unresolved']) or '-'}")
