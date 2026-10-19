"""selftest.py — the invariants. Offline, no data files, no API calls, seconds.

    python selftest.py

There is no dataset here, no calibration, no eval, and no measured accuracy.
Nothing in this project can tell you it got a diagnosis right. What this file
can tell you is that the rules the product is built on still hold — that the
panel cannot re-ask an answered question, cannot silently remove a warning,
cannot print a number, and cannot invent findings out of an empty
consultation.

Those are the rules in CLAUDE.md under "Non-negotiable". Every one of them is
non-negotiable because breaking it produced a real bug, a real fabrication, or
a real piece of false precision at some point in this project's history. This
file is what keeps them from coming back quietly.

WHAT THIS FILE DELIBERATELY DOES NOT TEST: whether the candidates are any
good. That needs a clinician scoring real consultations, and until that
happens no amount of green output here means the panel is useful. Do not read
a pass as a safety claim.
"""

import io
import json
import re
import runpy
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml

import engine
import extract
import propose
import vocabulary

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
_failures = []
_checks = 0


def check(name, condition, detail=""):
    global _checks
    _checks += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}{('  — ' + detail) if detail else ''}")
        _failures.append(name)


def section(title):
    print(f"\n{title}")


def browser_checks(scenario):
    """Execute the real frontend in Node with a fake DOM and no network."""
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    source = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    harness = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(0, 'utf8');
const elements = new Map();
function element() {
  const classes = new Set();
  return {
    innerHTML:'', textContent:'', value:'', disabled:false, style:{}, dataset:{},
    parentElement:{scrollTop:0}, appendChild(){}, addEventListener(){},
    querySelector(){return null;}, querySelectorAll(){return [];},
    classList:{add(c){classes.add(c);}, remove(c){classes.delete(c);},
      contains(c){return classes.has(c);},
      toggle(c,on){if(on===undefined)on=!classes.has(c);if(on)classes.add(c);else classes.delete(c);}}
  };
}
const document = {
  getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},
  createElement:element, addEventListener(){}
};
const context = vm.createContext({document,console,AbortController,
  setTimeout(){return 1;},clearTimeout(){},
  fetch(){throw new Error('Unexpected fetch in an offline browser test');}});
vm.runInContext(source, context);
const run = code => vm.runInContext(code, context);
(async()=>{
  const results = {};
  // SCENARIO
  process.stdout.write(JSON.stringify(results));
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
""".replace("// SCENARIO", scenario)
    try:
        result = subprocess.run(["node", "-e", harness], input=source, text=True, encoding="utf-8",
                                capture_output=True, cwd=ROOT, timeout=10)
    except FileNotFoundError:
        sys.exit("Node.js is required for the browser checks")
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


# A candidate set standing in for propose.py's output. Hand-written so these
# tests never touch the network: the invariants are properties of engine.py
# and the two YAML files, not of any model's behaviour on a given day.
FINDINGS = {
    "chest_tightness": True,
    "breathlessness": True,
    "onset_sudden_seconds": True,
    "palpitations": True,
    "anxiety_or_panic_feeling": True,
    "immobility_or_long_journey": True,
    "on_oral_contraceptive": True,
    "fever": False,
    "cough": False,
}

CANDIDATES = [
    {
        "condition": "Pulmonary embolism",
        "reasoning": "Sudden breathlessness with chest tightness, on a combined oral contraceptive, after a long journey.",
        "supported_by": ["breathlessness", "onset_sudden_seconds", "on_oral_contraceptive",
                         "immobility_or_long_journey", "chest_tightness"],
        "opposed_by": [],
        "unresolved": ["calf_swelling_unilateral", "pain_pleuritic", "haemoptysis"],
    },
    {
        "condition": "Panic attack",
        "reasoning": "Sudden chest tightness with palpitations and a feeling of fright.",
        "supported_by": ["anxiety_or_panic_feeling", "palpitations", "chest_tightness"],
        "opposed_by": [],
        "unresolved": ["pain_pleuritic", "prior_panic_attacks"],
    },
    {
        "condition": "Possible NSTEMI / STEMI",
        "reasoning": "Chest tightness of sudden onset with palpitations.",
        "supported_by": ["chest_tightness", "palpitations"],
        "opposed_by": [],
        "unresolved": ["chest_pain_radiates_left_arm_or_jaw", "diaphoresis"],
    },
    {
        # Absent from severity.yaml on purpose — the unrated tier must render.
        "condition": "Postural orthostatic tachycardia syndrome",
        "reasoning": "Palpitations and fatigue following a period of immobility.",
        "supported_by": ["palpitations"],
        "opposed_by": [],
        "unresolved": ["lightheaded_on_standing"],
    },
]

eng = engine.Engine.load()


# ===========================================================================
section("severity.yaml — the only human-auditable safeguard in the system")
# ===========================================================================
sev_raw = yaml.safe_load((ROOT / "severity.yaml").read_text(encoding="utf-8"))
pathologies = sev_raw["pathologies"]
excluded_by = sev_raw["excluded_by"]
eligible = {c for c, v in pathologies.items() if v is not None and v >= engine.CANT_MISS_FLOOR}

# If a condition can reach Can't-Miss it must be able to LEAVE Can't-Miss.
# Without this the panel answers "what haven't you ruled out" with a question
# that can never be finished, PE and ACS appear on every chest complaint
# forever, and the clinician stops reading it within a week. The previous
# file violated this for 19 of 37 eligible conditions, one of them at the top
# severity band.
missing_exit = sorted(eligible - set(excluded_by))
check("every panel-eligible condition has an exclusion rule", not missing_exit, str(missing_exit))

orphans = sorted(set(excluded_by) - set(pathologies))
check("no exclusion rule for a condition that does not exist", not orphans, str(orphans))

# `default: null`, not 3. An unlisted condition is UNRATED, not benign — the
# rulebook not knowing something is a fact about the rulebook, and must not be
# silently rendered as "probably fine".
check("default severity is null, so unlisted means unrated", sev_raw.get("default") is None)


# ===========================================================================
section("findings.yaml — stable ids are what make 'never re-ask' possible")
# ===========================================================================
find_raw = yaml.safe_load((ROOT / "findings.yaml").read_text(encoding="utf-8"))

seen, dupes, bad_id, incomplete = set(), [], [], []
for system, group in find_raw["findings"].items():
    for fid, meta in group.items():
        if fid in seen:
            dupes.append(fid)
        seen.add(fid)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", fid):
            bad_id.append(fid)
        if not (meta.get("label") and meta.get("ask")):
            incomplete.append(fid)

check("finding ids are unique across systems", not dupes, str(dupes))
check("finding ids are lowercase snake_case", not bad_id, str(bad_id))
check("every finding has a label and a question", not incomplete, str(incomplete))
check("vocabulary loaded the same ids", len(vocabulary.FINDINGS) == len(seen))

# Every exclusion token needs a human label. An unlabelled one would render as
# a raw identifier in the single prompt in this system that takes a warning
# off the screen.
tokens = {t for toks in excluded_by.values() for t in toks}
unlabelled = sorted(tokens - set(vocabulary.INVESTIGATIONS))
check("every exclusion action has a readable label", not unlabelled, str(unlabelled))
check("no investigation label nothing refers to",
      not sorted(set(vocabulary.INVESTIGATIONS) - tokens))


# ===========================================================================
section("the panel")
# ===========================================================================
out = eng.assess(CANDIDATES, FINDINGS)

# --- no number ever reaches the screen -------------------------------------
# Not a percentage, not a score, not a confidence. Nobody is ever certain in
# medicine, and three significant figures imply a precision no input to this
# system can support. Checked over the whole serialised payload, because the
# frontend renders from it and anything numeric in there can surface.
blob = json.dumps(out)
leaks = re.findall(r"\d+(?:\.\d+)?\s?%|\b\d+\.\d+\b", blob)
check("no percentage or decimal anywhere in the payload", not leaks, str(leaks))
check("no probability, score or confidence field",
      not any(k in blob for k in ('"p":', '"score"', '"confidence"', '"probability"', '"entropy"')))

# --- severity and support are two axes, never multiplied -------------------
row = out["cant_miss"][0]
check("severity and support are separate fields",
      "severity_band" in row and "support" in row)
check("severity band is a word, not a number", isinstance(row["severity_band"], str))
check("no blended risk field exists",
      not any(k in row for k in ("risk", "weighted", "priority_score")))

# --- the severity floor holds ----------------------------------------------
# A critical condition must never sort below a routine one, however well the
# findings support the routine one. Panic attack is better supported here than
# NSTEMI and must still not outrank it.
cm = [r["condition"] for r in out["cant_miss"]]
check("pulmonary embolism reaches Can't-Miss", "Pulmonary embolism" in cm, str(cm))
check("a self-limiting condition never reaches Can't-Miss", "Panic attack" not in cm)
check("panic attack is still shown, in the differential",
      "Panic attack" in [r["condition"] for r in out["differential"]])
bands = [r["severity_band"] for r in out["cant_miss"]]
check("Can't-Miss is ordered by severity band, worst first",
      bands == sorted(bands, key=lambda b: [x[1] for x in engine.BANDS].index(b)), str(bands))

# --- a naming variant must not silently demote a critical condition --------
# The candidate list is unbounded, so conditions arrive named however the
# model named them. Exact string matching against severity.yaml meant a
# spelling could push a critical condition off Can't-Miss and take its
# exclusion rule with it — a warning removed by a string comparison, which is
# the one thing this system must never do. Observed for real on the canonical
# case: "Panic attack (acute anxiety episode)" scored unrated.
for variant, expected in [
    ("Acute pulmonary embolism", "Pulmonary embolism"),
    ("Pulmonary embolism (PE)", "Pulmonary embolism"),
    ("Subarachnoid hemorrhage", "Subarachnoid haemorrhage"),   # American spelling
    ("Severe anemia", "Severe anaemia"),
    ("Panic attack (acute anxiety episode)", "Panic attack"),
    ("NSTEMI", "Possible NSTEMI / STEMI"),                     # one side of an alternation
    ("TIA", "Acute stroke / TIA"),
    ("Suspected bacterial meningitis", "Bacterial meningitis"),
    # The rulebook being the MORE specific of the two. Containment cannot
    # catch these, and both were scoring unrated on a real consultation —
    # dropping a critical condition off Can't-Miss and losing its exclusion
    # rule. They match on a word unique to one rulebook entry.
    ("Meningococcal meningitis", "Bacterial meningitis"),
    ("Dengue fever", "Dengue with warning signs"),
    ("Viral gastroenteritis", "Gastroenteritis"),
    ("Bacterial pneumonia", "Pneumonia"),
    # Acronym against expansion. severity.yaml writes some entries short and
    # a model writes them long; the two share no words at all, so "Systemic
    # lupus erythematosus flare" scored unrated against an entry named "SLE".
    ("Systemic lupus erythematosus flare", "SLE"),
    ("Paroxysmal supraventricular tachycardia", "PSVT"),
    ("Upper respiratory tract infection", "URTI"),
    ("SAH", "Subarachnoid haemorrhage"),
    ("DKA", "Diabetic ketoacidosis"),
    ("UTI", "Urinary tract infection"),
    ("Giant cell arteritis", "Temporal arteritis"),
    ("Right ovarian torsion", "Ovarian torsion"),
]:
    found = eng.match(variant)
    check(f"'{variant}' is rated as '{expected}'",
          found is not None and found[0] == expected, str(found))

# Generous, but not credulous. A condition genuinely absent from the rulebook
# must stay unrated rather than being mapped onto whatever looks nearest.
for stranger in ["Costochondritis", "Vitamin D deficiency",
                 "Postural orthostatic tachycardia syndrome",
                 # A word can be unique to one rulebook entry by accident and
                 # still identify nothing. "Viral exanthem" matched "Viral
                 # pharyngitis" on `viral` alone — rating a rash as a sore
                 # throat — and "Stevens-Johnson syndrome / drug reaction"
                 # matched "Acute dystonic reaction" on `reaction`.
                 "Viral exanthem", "Stevens-Johnson syndrome / drug reaction",
                 "Gastric neoplasm",
                 # Anatomical and positional words identify a SITE, not a
                 # condition. Every one of these matched something before the
                 # generic list caught it, and the last three landed at the
                 # top severity band: "Lower respiratory tract infection" as a
                 # urinary infection, "Upper respiratory tract infection" as
                 # an upper GI bleed, "Diabetic neuropathy" as ketoacidosis,
                 # "Thyroid nodule" as thyroid storm.
                 "Postural orthostatic tachycardia syndrome",
                 "Lower respiratory tract infection", "Systemic sclerosis",
                 "Sinus tachycardia", "Temporal lobe epilepsy",
                 "Urinary retention", "Left ventricular failure",
                 "Diabetic neuropathy", "Thyroid nodule", "Ovarian cyst",
                 "Molar pregnancy", "Cerebral palsy"]:
    check(f"'{stranger}' is honestly unrated", eng.match(stranger) is None)

# Where several entries match, the most severe wins — same safe direction.
check("a compound name takes the more dangerous of its matches",
      eng.match("Sepsis secondary to pneumonia")[0] == "Sepsis")

# Aliases name the SAME condition; they must never invent a severity for one
# the rulebook does not have, and every alias must point at a real entry.
_aliases = sev_raw.get("aliases") or {}
check("every alias points at a condition that exists",
      not sorted(set(_aliases) - set(pathologies)),
      str(sorted(set(_aliases) - set(pathologies))))

# A matched condition must still be able to LEAVE the panel. Reaching
# Can't-Miss under a matched name but finding no exclusion rule under the
# name the model used would show something that can never be cleared.
check("a matched condition keeps its exclusion rule",
      len(eng.exclusion_options("Acute pulmonary embolism")) > 0)

# And the substitution must be visible. The one place a generous match could
# go wrong is the one place the clinician has to be able to see it.
variant_out = eng.assess([dict(CANDIDATES[1], condition="Panic attack (acute anxiety episode)")],
                         FINDINGS)
check("the rulebook entry actually applied is shown when it differs",
      variant_out["differential"][0]["rated_as"] == "Panic attack")
check("an exact match adds no redundant note",
      out["cant_miss"][0]["rated_as"] is None)

# --- severity-unrated is a visible tier ------------------------------------
# A proposed condition absent from severity.yaml still appears, marked
# unrated. It is never silently dropped, and it must not crash the renderer.
unrated = [r["condition"] for r in out["unrated"]]
check("a condition missing from severity.yaml renders in its own tier",
      "Postural orthostatic tachycardia syndrome" in unrated, str(unrated))
check("the unrated tier says so in words", out["unrated"][0]["severity_band"] == "unrated")

# --- every candidate carries its reasoning ---------------------------------
# With no eval, the visible reasoning and the clinician ARE the entire safety
# net. A suggestion whose justification cannot be read cannot be checked.
every_row = out["cant_miss"] + out["differential"] + out["unrated"]
check("every candidate on the panel carries readable reasoning",
      all(r.get("reasoning", "").strip() for r in every_row))
check("no reasoning text contains a percentage",
      not any(re.search(r"\d+\s?%", r["reasoning"]) for r in every_row))


# ===========================================================================
section("never re-ask an answered finding")
# ===========================================================================
q = out["next_question"]
check("a question is offered", q is not None)
check("the question divides the candidate list", len(q["divides"]) >= 1)
check("the next question is about the leg",
      q["finding"] in ("calf_swelling_unilateral", "calf_pain_or_tenderness"), q["finding"])

# Answered means answered, in EITHER direction. The previous build could not
# match a whole-question denial against its own asked set, so the identical
# question came back every single turn. This is the regression test for that.
answered_present = dict(FINDINGS, calf_swelling_unilateral=True)
answered_absent = dict(FINDINGS, calf_swelling_unilateral=False)
q_present = eng.assess(CANDIDATES, answered_present)["next_question"]
q_absent = eng.assess(CANDIDATES, answered_absent)["next_question"]
check("a finding answered PRESENT is never asked again",
      q_present is None or q_present["finding"] != "calf_swelling_unilateral")
check("a finding answered ABSENT is never asked again",
      q_absent is None or q_absent["finding"] != "calf_swelling_unilateral")

# Asked-but-unanswered also counts. The patient was put the question; asking
# it again because they did not answer is the same failure with extra steps.
q_asked = eng.assess(CANDIDATES, FINDINGS, asked=["calf_swelling_unilateral"])["next_question"]
check("a finding already asked is not asked again",
      q_asked is None or q_asked["finding"] != "calf_swelling_unilateral")

# Repeatedly answering must eventually exhaust the questions rather than loop.
state, seen_q, loops = dict(FINDINGS), [], 0
while loops < 25:
    nq = eng.assess(CANDIDATES, state)["next_question"]
    if nq is None:
        break
    seen_q.append(nq["finding"])
    state[nq["finding"]] = False
    loops += 1
check("the question sequence terminates", loops < 25, f"still asking after {loops}")
check("no question is ever repeated", len(seen_q) == len(set(seen_q)), str(seen_q))

# Examination items are legitimate next steps but must never be spoken to the
# patient, and the caller cannot tell that from the question text alone.
check("clinician-only questions are flagged as such",
      eng.assess([{"condition": "Cardiac tamponade", "reasoning": "x",
                   "supported_by": ["breathlessness"], "opposed_by": [],
                   "unresolved": ["heart_sounds_muffled"]}],
                 {"breathlessness": True})["next_question"]["clinician_only"] is True)


# ===========================================================================
section("exclusion proposes; it never auto-removes")
# ===========================================================================
# This is the one failure mode in this system shaped like patient harm rather
# than noise. Every other error adds something to a list a human is reading;
# an exclusion error REMOVES a warning, and nobody ever sees what they were
# not shown. So a human makes every removal.
pe = next(r for r in out["cant_miss"] if r["condition"] == "Pulmonary embolism")
check("a Can't-Miss entry offers the actions that would exclude it",
      len(pe.get("exclusion_options", [])) > 0)
check("exclusion actions are shown with readable labels",
      all(o["label"] and o["label"] != o["action"] for o in pe["exclusion_options"]),
      str(pe["exclusion_options"]))

# Recording the action alone changes nothing. Only a clinician's confirmation
# does, and it arrives as confirmed_exclusions on the next call.
after = eng.assess(CANDIDATES, FINDINGS, confirmed_exclusions=["Pulmonary embolism"])
check("a confirmed exclusion leaves Can't-Miss",
      "Pulmonary embolism" not in [r["condition"] for r in after["cant_miss"]])
check("but is still shown as ruled out, not deleted",
      "Pulmonary embolism" in [r["condition"] for r in after["ruled_out"]])
check("its reasoning survives exclusion, so the decision stays auditable",
      after["ruled_out"][0]["reasoning"].strip() != "")
check("an excluded condition no longer drives the next question",
      after["next_question"] is None
      or "Pulmonary embolism" not in after["next_question"]["divides"])

renamed_pe = dict(CANDIDATES[0], condition="Acute pulmonary embolism")
renamed_exclusion = eng.assess([renamed_pe], FINDINGS,
                              confirmed_exclusions=["Pulmonary embolism"])
check("a confirmed exclusion covers severity-matched naming variants",
      not renamed_exclusion["cant_miss"] and len(renamed_exclusion["ruled_out"]) == 1)
check("an excluded naming variant does not drive the next question",
      renamed_exclusion["next_question"] is None)
reverse_exclusion = eng.assess([CANDIDATES[0]], FINDINGS,
                              confirmed_exclusions=["Acute pulmonary embolism"])
check("confirmation under a variant also excludes the canonical name",
      not reverse_exclusion["cant_miss"] and len(reverse_exclusion["ruled_out"]) == 1)
missing_proposal = eng.assess([CANDIDATES[1]], FINDINGS,
                             confirmed_exclusions=["Pulmonary embolism"])
check("a confirmed exclusion stays auditable when no longer proposed",
      any(r["condition"] == "Pulmonary embolism" and r["reasoning"]
          for r in missing_proposal["ruled_out"]))
audit_only = eng.assess([], FINDINGS, confirmed_exclusions=["Pulmonary embolism"])
check("an audit-only consultation still shows its confirmed exclusions",
      not audit_only["abstaining"] and len(audit_only["ruled_out"]) == 1
      and not audit_only["cant_miss"] and audit_only["next_question"] is None)
check("a naming variant without clinician confirmation stays open",
      len(eng.assess([renamed_pe], FINDINGS)["cant_miss"]) == 1)


# ===========================================================================
section("abstain visibly; fabricate nothing")
# ===========================================================================
# When nothing was extracted, say so. Never show a generic suggestion styled
# as though it came from this patient. Both of these run with no network at
# all — the abstain paths must not cost a model call.
empty_findings, empty_quotes = extract.extract("", "")
check("an empty consultation extracts nothing", empty_findings == {} and empty_quotes == {})
check("nothing present proposes nothing", propose.propose({"fever": False}) == [])

nothing = eng.assess([], {})
check("no candidates means abstaining", nothing["abstaining"] is True)
check("abstaining says so in plain words", bool(nothing["message"]))
check("abstaining shows no candidates anywhere",
      not (nothing["cant_miss"] or nothing["differential"] or nothing["unrated"]))
check("abstaining offers no question it cannot justify", nothing["next_question"] is None)

# The message must not tell the reader to answer questions that are not there.
# It did exactly that once: the abstain text said "ask the questions below"
# while the question panel showed that same message back.
check("the abstain message does not point at absent questions",
      "below" not in (nothing["message"] or "").lower(), nothing["message"])

# A model claiming a finding supports a condition when the patient does not
# have that finding is asserting evidence that does not exist — the one thing
# a clinician skim-reading the panel would not catch. propose.py filters those
# out; engine.py must survive one arriving anyway, because a crash in triage
# shows an empty panel and an empty panel reads as "nothing to worry about".
malformed = [
    {"condition": "Aortic dissection", "reasoning": "test",
     "supported_by": ["haemoptysis", {"nested": "junk"}, None],
     "opposed_by": None, "unresolved": ["not_a_real_finding_id"]},
]
try:
    bogus = eng.assess(malformed, FINDINGS)
    ok = len(bogus["cant_miss"]) == 1 and bogus["next_question"] is None
except Exception as e:
    ok, bogus = False, str(e)
check("a malformed candidate renders instead of crashing the panel", ok, str(bogus))


# ===========================================================================
section("model output validation")
# All nonempty extraction/screening responses are mocked: never call a model.
def extracted_response(response):
    with patch.object(extract.llm, "cached", return_value=response):
        try:
            return extract.extract("offline test input"), None
        except Exception as exc:
            return ({}, {}), str(exc)

validated, error = extracted_response({
    "present": [{"id": "fever", "evidence": "  "}],
    "absent": [{"id": "cough", "evidence": ""}],
})
check("findings with empty evidence quotes are dropped", not error and validated == ({}, {}), str(validated))
validated, error = extracted_response({
    "present": [{"id": "fever", "evidence": "fever"}, {"id": "cough", "evidence": "cough"}],
    "absent": [{"id": "fever", "evidence": "no fever"}],
})
check("conflicting present and absent IDs become unknown with no quote",
      not error and validated == ({"cough": True}, {"cough": "cough"}), str(validated))
validated, error = extracted_response({
    "present": [{"id": "fever", "evidence": "fever"}],
    "absent": [{"id": "fever", "evidence": ""}],
})
check("a conflicting ID is unknown even when one quote is empty",
      not error and validated == ({}, {}), str(validated))
validated, error = extracted_response({
    "present": [None, "fever", 7, [], {}, {"id": 7, "evidence": "x"},
                {"id": "fever", "evidence": ["x"]}, {"id": "cough", "evidence": "cough"}],
    "absent": {"id": "fever", "evidence": "no fever"},
})
check("malformed extraction entries are skipped while valid entries survive",
      not error and validated == ({"cough": True}, {"cough": "cough"}), error or str(validated))
validated, error = extracted_response(["not an extraction object"])
check("a malformed extraction response does not crash",
      not error and validated == ({}, {}), error or str(validated))
for raw, expected_clinical in [("false", False), (" FALSE ", False), ("true", True), (False, False), (True, True)]:
    with patch.object(extract.llm, "cached", return_value={"clinical": raw, "kind": "test document"}):
        screened = extract.screen_document("offline OCR test")
    check(f"document screening parses {raw!r} as {expected_clinical}", screened["clinical"] is expected_clinical)
with patch.object(extract.llm, "cached", return_value=["malformed screen response"]):
    try:
        screened, error = extract.screen_document("offline OCR test"), None
    except Exception as exc:
        screened, error = {}, str(exc)
check("malformed screening output keeps the document visibly unscreened",
      not error and screened == {"clinical": True, "kind": "not screened"}, error or str(screened))


# ===========================================================================
section("candidate reasoning does not invent certainty")
withheld_reasoning = "Reasoning withheld: the model used numeric or probability wording."


def reasoning_payload_is_clean(candidates):
    payload = json.dumps(eng.assess(candidates, FINDINGS), ensure_ascii=False)
    # The fixed withholding notice explains the removal; it is not a model claim.
    model_text = payload.replace(withheld_reasoning, "")
    return not re.search(
        r"[%％]|\b(?:percent(?:age)?|per\s+cent|probabilit(?:y|ies)|likelihood|confidence)\b"
        r"|\bchance\s+of\b|\bp\s*(?:=|\()|\b\d+\.\d+\b", model_text, re.I)


for unsafe_reasoning in ["A 95% chance of PE", "The probability of PE is high.",
                         "There is a chance of PE.", "Ninety-five percent certainty.",
                         "P(PE) = 0.95.", "The likelihood of PE is high."]:
    response = {"candidates": [dict(CANDIDATES[0], reasoning=unsafe_reasoning), CANDIDATES[1]]}
    with patch.object(propose.llm, "cached", return_value=response):
        checked_candidates = propose.propose(FINDINGS)
    check(f"unsafe reasoning is withheld without losing its candidate: {unsafe_reasoning}",
          checked_candidates == [dict(CANDIDATES[0], reasoning=withheld_reasoning), CANDIDATES[1]])
    check(f"unsupported claim cannot reach the payload: {unsafe_reasoning}",
          reasoning_payload_is_clean(checked_candidates))

mixed_reasoning = "A 95% chance of PE. Sudden breathlessness after a long journey."
with patch.object(propose.llm, "cached", return_value={
        "candidates": [dict(CANDIDATES[0], reasoning=mixed_reasoning)]}):
    checked_candidates = propose.propose(FINDINGS)
check("only the unsupported sentence is removed, preserving candidate evidence",
      checked_candidates == [dict(CANDIDATES[0], reasoning="Sudden breathlessness after a long journey.")])
check("mixed reasoning cannot leak a percentage or probability claim into the payload",
      reasoning_payload_is_clean(checked_candidates))

with patch.object(propose.llm, "cached", return_value={"candidates": [CANDIDATES[0]]}):
    checked_candidates = propose.propose(FINDINGS)
check("factual candidate reasoning survives validation",
      len(checked_candidates) == 1 and checked_candidates[0]["reasoning"] == CANDIDATES[0]["reasoning"])


# ===========================================================================
section("browser checks require Node.js")
with patch.object(subprocess, "run", side_effect=FileNotFoundError("node is not installed")):
    try:
        browser_checks("")
        missing_node_exit = None
    except (SystemExit, Exception) as exc:
        missing_node_exit = exc
check("missing Node.js exits with a clear prerequisite message",
      isinstance(missing_node_exit, SystemExit)
      and missing_node_exit.code == "Node.js is required for the browser checks",
      str(missing_node_exit))


# ===========================================================================
section("patient details are rendered as text")
try:
    browser_result = browser_checks(r"""
run(`SETUP={name:'<img src=x onerror="alert(1)"> & Alice',age:'34',gender:'F',
  caseId:'<svg onload="alert(2)">',languageCode:'en-IN'};paintPatientBar();`);
const header = elements.get('patientbar').innerHTML;
results.escaped_name = header.includes('&lt;img') && header.includes('&amp; Alice') && !header.includes('<img');
results.escaped_case_id = header.includes('&lt;svg') && header.includes('&quot;alert(2)&quot;') && !header.includes('<svg');
""")
    browser_error = ""
except Exception as exc:
    browser_result, browser_error = {}, str(exc)
check("patient name cannot insert HTML into the header", browser_result.get("escaped_name", False), browser_error)
check("case ID cannot insert HTML into the header", browser_result.get("escaped_case_id", False), browser_error)


# ===========================================================================
section("analysis results follow the current consultation and evidence")
try:
    browser_result = browser_checks(r"""
function stage(marker){return {abstaining:false,message:null,
  cant_miss:[{condition:marker,reasoning:'offline fixture',severity_band:'critical',support:'limited',
    supported_by:[],opposed_by:[],exclusion_options:[]}],
  differential:[],unrated:[],ruled_out:[],doesnt_fit:[],next_question:null};}
function prime(){
  run('reset()');context.priorStage=stage('old-result');
  run(`dxState={stage:priorStage};cmState={stage:priorStage};committed=true;hasLiveInput=true;
    liveTranscriptExtra='offline evidence';paintDX();paintCM();paintQB();paintMIS();`);
}
function response(data){return {ok:true,json:async()=>data};}
prime();run(`liveTranscriptExtra='';scannedDocs=[{use:false,text:'removed document'}];`);
await run('runAnalyze()');
results.empty_clears = run('dxState===null && cmState===null') && !elements.get('cm').innerHTML.includes('old-result');
prime();run(`liveTranscriptExtra='  ';scannedDocs=[];`);await run('runAnalyze()');
results.whitespace_clears = run('dxState===null && cmState===null');

prime();let finishOld;
context.fetch=()=>new Promise(resolve=>{finishOld=resolve;});
const oldRequest=run('runAnalyze()');run('reset()');
results.reset_unlocks = run('analyzeInFlight===false');
finishOld(response(stage('old-response')));await oldRequest;
results.reset_ignores_response = run('dxState===null && cmState===null');

prime();const finishes=[];
context.fetch=()=>new Promise(resolve=>finishes.push(resolve));
const older=run('runAnalyze()');run(`reset();committed=true;hasLiveInput=true;liveTranscriptExtra='new evidence';`);
const newer=run('runAnalyze()');
finishes[0](response(stage('old-response')));await older;
results.old_request_does_not_unlock_new = finishes.length===2 && run('analyzeInFlight===true && dxState===null');
if(finishes[1])finishes[1](response(stage('new-response')));await newer;
results.new_response_wins = run(`dxState && dxState.stage.cant_miss[0].condition==='new-response'`);

prime();let finishRemoved;
context.fetch=()=>new Promise(resolve=>{finishRemoved=resolve;});
const removedRequest=run('runAnalyze()');run(`liveTranscriptExtra='';scannedDocs=[];`);await run('runAnalyze()');
finishRemoved(response(stage('removed-evidence-result')));await removedRequest;
results.removed_response_ignored = run('dxState===null && cmState===null');

prime();context.fetch=async()=>({ok:false,status:502,json:async()=>({detail:'provider unavailable'})});
await run('runAnalyze()');
results.http_failure_visible = document.getElementById('analysis-error').textContent.includes('provider unavailable')
  && document.getElementById('analysis-error').textContent.toLowerCase().includes('stale')
  && elements.get('cmtag').textContent.toLowerCase().includes('stale')
  && elements.get('cm').innerHTML.includes('old-result');
context.fetch=async()=>response(stage('recovered-result'));await run('runAnalyze()');
results.recovery_clears_error = document.getElementById('analysis-error').textContent===''
  && !elements.get('cmtag').textContent.toLowerCase().includes('stale')
  && elements.get('cm').innerHTML.includes('recovered-result');
prime();context.fetch=async()=>{throw new Error('connection lost');};await run('runAnalyze()');
results.network_failure_visible = document.getElementById('analysis-error').textContent.includes('connection lost')
  && document.getElementById('analysis-error').textContent.toLowerCase().includes('stale');
""")
    browser_error = ""
except Exception as exc:
    browser_result, browser_error = {}, str(exc)
for key, description in [
    ("empty_clears", "removing all included evidence clears old results"),
    ("whitespace_clears", "whitespace-only evidence also clears results"),
    ("reset_unlocks", "reset allows a new analysis without waiting for the old request"),
    ("reset_ignores_response", "analysis responses from before reset are ignored"),
    ("old_request_does_not_unlock_new", "an obsolete request cannot unlock a newer request"),
    ("new_response_wins", "the new consultation receives its own analysis"),
    ("removed_response_ignored", "removed evidence cannot return via an in-flight response"),
    ("http_failure_visible", "HTTP failures retain old results with a visible stale warning"),
    ("recovery_clears_error", "successful analysis clears the previous error and stale marker"),
    ("network_failure_visible", "network failures are visible instead of silently reverting"),
]:
    check(description, browser_result.get(key, False), browser_error)


# ===========================================================================
section("proposal cache invalidation includes finding labels")
def proposal_cache_key():
    module = runpy.run_path(str(ROOT / "propose.py"), run_name="offline_cache_test")
    keys = []

    def cached_result(path, produce):
        keys.append(str(path))
        return {"candidates": []}

    with patch.object(propose.llm, "cached", side_effect=cached_result):
        module["propose"]({"headache": True})
    return keys[0]

original_key = proposal_cache_key()
with patch.dict(vocabulary.FINDINGS["headache"], {"label": "Changed label for an offline cache test"}):
    changed_key = proposal_cache_key()
check("changing only a finding label invalidates proposal responses", original_key != changed_key)
check("unchanged finding labels retain the same proposal cache key", original_key == proposal_cache_key())


# ===========================================================================
section("unrated logging contains no consultation evidence")
import main

with tempfile.TemporaryDirectory(dir=ROOT / "cache", prefix="selftest-unrated-") as log_test_dir:
    log_path = Path(log_test_dir) / "cache" / "unrated_log.jsonl"
    request = main.AnalyzeRequest(transcript="PRIVATE_TRANSCRIPT", doc_text="PRIVATE_DOCUMENT")
    with patch.object(main, "REPO_ROOT", Path(log_test_dir)), \
            patch.object(main.extract, "extract", return_value=(FINDINGS, {"breathlessness": "PRIVATE_QUOTE"})), \
            patch.object(main.propose, "propose", return_value=[CANDIDATES[-1]]):
        logged_payload = main._pipeline(request)
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] if log_path.exists() else []
        check("an unrated log line contains only timestamp and condition",
              len(records) == 1 and set(records[0]) == {"timestamp", "condition"}
              and records[0]["condition"] == CANDIDATES[-1]["condition"]
              and "PRIVATE_" not in json.dumps(records), str(records))
        valid_timestamp = False
        if records:
            try:
                valid_timestamp = datetime.fromisoformat(records[0]["timestamp"]).tzinfo is not None
            except (KeyError, TypeError, ValueError):
                pass
        check("the unrated log timestamp is timezone-aware ISO format", valid_timestamp)
        main._pipeline(request)
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] if log_path.exists() else []
        check("unrated conditions append one private-safe line per analysis",
              len(records) == 2 and all(set(r) == {"timestamp", "condition"} for r in records))
        log_error = io.StringIO()
        with patch.object(Path, "open", side_effect=PermissionError("offline log write failure")), \
                patch("sys.stderr", log_error):
            try:
                failed_log_payload, error = main._pipeline(request), None
            except Exception as exc:
                failed_log_payload, error = {}, str(exc)
        check("a log write failure is reported without hiding the clinical panel",
              not error and bool(failed_log_payload.get("unrated")) and "log" in log_error.getvalue().lower(), error or log_error.getvalue())


# ===========================================================================
section("determinism — identical input, identical output")
# ===========================================================================
# Confirmed live once: identical input returned different findings run to run
# until temperature and seed were pinned, which changed what was on screen for
# reasons that had nothing to do with the patient. engine.py has no randomness
# at all and must stay that way.
a = json.dumps(eng.assess(CANDIDATES, FINDINGS), sort_keys=True)
b = json.dumps(eng.assess(CANDIDATES, FINDINGS), sort_keys=True)
check("triage is deterministic", a == b)

shuffled = list(reversed(CANDIDATES))
c = json.dumps(eng.assess(shuffled, FINDINGS), sort_keys=True)
check("candidate order from propose.py does not change the panel", a == c)


# ===========================================================================
section("the case fixtures are expressible")
# ===========================================================================
# Without this, nothing defines what correct output looks like and there is
# nothing for a regression to be measured against.
case_paths = sorted(p for p in (ROOT / "cases").glob("*.yaml") if not p.name.startswith("_"))
if not case_paths:
    raise RuntimeError("No case fixtures found in cases/ (underscore-prefixed files are skipped)")
for case_path in case_paths:
    case = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    print(f"  case: {case['case']['id']}")
    ef = case["expected_findings"]
    refs = ef["present"] + ef["absent"] + ef["must_be_unknown"] + case["expectations"]["next_question_in"]
    unknown_refs = sorted(f for f in refs if f not in vocabulary.FINDINGS)
    check("every finding the case names exists in the vocabulary",
          not unknown_refs, str(unknown_refs))
    check("pulmonary embolism is rated at the top severity band",
          eng.band("Pulmonary embolism")[0] == "critical")
    check("pulmonary embolism can be excluded", len(eng.exclusion_options("Pulmonary embolism")) > 0)
    check("the doctor's own working diagnosis is rated, not unrated",
          eng.band("Panic attack")[0] != "unrated")

pe_fixture = ROOT / "cases" / "pe_masquerading_as_panic.yaml"
check("the PE fixture's first comment uses its current filename",
      pe_fixture.read_text(encoding="utf-8").splitlines()[0].startswith("# pe_masquerading_as_panic.yaml"))


# ===========================================================================
print(f"\n{'=' * 70}")
if _failures:
    print(f"{len(_failures)} of {_checks} FAILED: {', '.join(_failures)}")
    sys.exit(1)
print(f"all {_checks} invariants hold.")
print("This says the rules still hold. It does NOT say the panel is useful —")
print("that needs a clinician scoring real consultations, and severity.yaml")
print("still has not been reviewed by one.")
