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

import asyncio
import base64
import io
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import yaml

import engine
import extract
import llm
import propose
import vocabulary
import capture

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
    contains(target){return target===this;},
    getBoundingClientRect(){return {left:100,top:200,right:400,bottom:500,width:300,height:300};},
    querySelector(){return null;}, querySelectorAll(){return [];},
    classList:{add(c){classes.add(c);}, remove(c){classes.delete(c);},
      contains(c){return classes.has(c);},
      toggle(c,on){if(on===undefined)on=!classes.has(c);if(on)classes.add(c);else classes.delete(c);}}
  };
}
const documentListeners = new Map();
const document = {
  getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},
  createElement:element,
  addEventListener(type, handler, capture=false){
    if(!documentListeners.has(type))documentListeners.set(type,[]);
    documentListeners.get(type).push({handler,capture});
  }
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
section("structured chat returns data instead of spending the output budget thinking")


def offline_chat(reply):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        message, finish_reason = reply(body, len(requests))
        return httpx.Response(200, json={
            "id": "offline-chat", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        sdk = llm.SarvamAI(api_subscription_key="offline-test-key", httpx_client=transport)
        with patch.object(llm, "client", return_value=sdk), patch.object(llm.time, "sleep"), \
                patch.dict(os.environ, {"SARVAM_SCREEN_MODEL": "sarvam-105b"}):
            try:
                result = llm.call_tool("offline system", "offline input", extract._SCREEN_TOOL, task="screen", max_tokens=256)
                error = None
            except Exception as exc:
                result, error = None, exc
    return result, requests, error


screen_arguments = {"clinical": True, "kind": "offline document"}
screen_reply = {"role": "assistant", "tool_calls": [{
    "id": "offline-tool", "type": "function", "function": {
        "name": "screen_document", "arguments": json.dumps(screen_arguments),
    },
}]}
missing_reply = {"role": "assistant", "content": None, "reasoning_content": "offline thinking"}

result, chat_requests, error = offline_chat(lambda body, attempt:
    (screen_reply, "tool_calls") if "reasoning_effort" in body and body["reasoning_effort"] is None
    else (missing_reply, "length"))
check("the SDK explicitly sends null reasoning_effort to disable hidden thinking",
      bool(chat_requests) and all("reasoning_effort" in r and r["reasoning_effort"] is None for r in chat_requests))
check("a short structured screening response succeeds without consuming retries",
      not error and result == screen_arguments and len(chat_requests) == 1, str(error))

result, chat_requests, error = offline_chat(lambda body, attempt:
    (missing_reply, "length") if attempt == 1 else (screen_reply, "tool_calls"))
check("missing tool calls still retry with a changed seed and the original forced schema",
      not error and result == screen_arguments and [r["seed"] for r in chat_requests] == [0, 1]
      and all(r["temperature"] == 0 and r["max_tokens"] == 256
              and r["tools"] == [extract._SCREEN_TOOL]
              and r["tool_choice"] == {"type": "function", "function": {"name": "screen_document"}}
              for r in chat_requests), str(error))
result, chat_requests, error = offline_chat(lambda body, attempt: (missing_reply, "length"))
check("exhausted tool-call retries remain an error rather than an empty clinical result",
      isinstance(error, llm.ToolCallMissing) and result is None and len(chat_requests) == 2)


# ===========================================================================
section("model output validation")
# All nonempty extraction/screening responses are mocked: never call a model.

# This checks the request contract, not whether a live model obeys the prompt.
# The live failure returned no findings for supplied sample records while the
# transcript was empty, despite OCR and screening both succeeding.
sample_document = (
    "Desogestrel 0.15 mg and Ethinylestradiol 0.03 mg Tablets IP\n"
    "SAMPLE - SYNTHETIC DOCUMENT FOR APP TESTING ONLY"
)
sample_quote = "Desogestrel 0.15 mg and Ethinylestradiol 0.03 mg Tablets IP"
with patch.object(llm, "cached", side_effect=lambda path, produce: produce()), \
        patch.object(llm, "call_tool", return_value={
            "present": [{"id": "on_oral_contraceptive", "evidence": sample_quote}],
            "absent": [],
        }) as sample_call:
    sample_findings, sample_quotes = extract.extract("", sample_document)
sample_system, sample_input, sample_tool = sample_call.call_args.args
check("document-only extraction sends the supplied record to the extraction model",
      sample_call.call_count == 1 and sample_document in sample_input
      and "TRANSCRIPT:\n\n\nDOCUMENTS" in sample_input
      and sample_tool == extract._TOOL and sample_call.call_args.kwargs == {"task": "extract"})
check("extraction abstains only when both transcript and documents lack clinical content",
      "If neither the transcript nor the supplied documents contains clinical content" in sample_system
      and "If the transcript contains no clinical content at all, record nothing" not in sample_system)
check("extraction explicitly retains evidence in sample records without relaxing quotation requirements",
      "sample or synthetic labels do not erase the clinical facts" in sample_system
      and "using the same evidence requirements" in sample_system
      and sample_findings == {"on_oral_contraceptive": True}
      and sample_quotes == {"on_oral_contraceptive": sample_quote})

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
section("supplemental findings have source-backed quotes and stay separate")


def extracted_other(response, transcript="My eyelid keeps twitching.", document=""):
    output = io.StringIO()
    with patch.object(llm, "cached", return_value=response), patch("sys.stdout", output):
        result = extract.extract(transcript, document, include_other=True)
    return result, output.getvalue()


supplemental_response = {"present": [], "absent": [], "other": [
    {"text": "Eyelid twitching", "evidence": "My eyelid keeps twitching."},
]}
supplemental, supplemental_log = extracted_other(supplemental_response)
check("a verbatim supplemental quote is kept outside the catalog dict and quotes",
      supplemental == ({}, {}, [{"id": "other:0", "text": "Eyelid twitching",
                                "evidence": "My eyelid keeps twitching."}]))
check("the supplemental schema caps entries and requires text and quoted evidence",
      extract._TOOL["function"]["parameters"]["properties"]["other"]["maxItems"] == 10
      and extract._TOOL["function"]["parameters"]["properties"]["other"]["items"]["required"] == ["text", "evidence"]
      and "Never put a denial" in extract.SYSTEM_PROMPT)
normalised, _ = extracted_other({"other": [{"text": "Eyelid twitching", "evidence": "MY eyelid\nkeeps twitching."}]},
                                 document="My  eyelid keeps\t twitching.", transcript="")
check("supplemental quote verification normalises only whitespace and case and reads documents",
      len(normalised[2]) == 1 and normalised[2][0]["evidence"] == "MY eyelid\nkeeps twitching.")
unquoted, unquoted_log = extracted_other({"other": [{"text": "Eyelid twitching", "evidence": "Her eyelid twitches."}]})
check("a paraphrased supplemental quote is dropped and its generated text is logged",
      unquoted[2] == [] and "Eyelid twitching" in unquoted_log and "not found verbatim" in unquoted_log)
cross_source, _ = extracted_other({"other": [{"text": "Eyelid twitching", "evidence": "eyelid twitching"}]},
                                  transcript="eyelid", document="twitching")
check("a supplemental quote cannot be assembled across transcript and document boundaries", cross_source[2] == [])
for unsafe in ["95% chance of eyelid twitching", "Probability of twitching", "High likelihood",
               "High confidence", "A chance of twitching", "Ten percent", "P(twitching)"]:
    rejected, rejection_log = extracted_other({"other": [{"text": unsafe, "evidence": "My eyelid keeps twitching."}]})
    check(f"supplemental generated probability wording is dropped and logged: {unsafe}",
          rejected[2] == [] and unsafe in rejection_log and "probability wording" in rejection_log)
numeric_source = "My eyelid twitches 5 times; a note says 95% probability."
numeric_other, _ = extracted_other({"other": [{"text": "Eyelid twitches 5 times", "evidence": numeric_source}]},
                                   transcript=numeric_source)
check("digits in supplemental text and probability wording in verbatim quotes are retained",
      numeric_other[2] == [{"id": "other:0", "text": "Eyelid twitches 5 times", "evidence": numeric_source}])
malformed_other, malformed_log = extracted_other({"other": [None, {}, {"text": 9, "evidence": []},
    {"text": " ", "evidence": "My eyelid keeps twitching."},
    {"text": " ".join(["word"] * 13), "evidence": "My eyelid keeps twitching."},
    supplemental_response["other"][0]]})
check("malformed and overlong supplemental entries do not crash or displace valid entries",
      malformed_other[2] == supplemental[2] and "dropped other" in malformed_log)
capped_other, capped_log = extracted_other({"other": [dict(supplemental_response["other"][0], text=f"Eyelid twitching {n}")
                                                       for n in range(11)]})
check("supplemental entries are capped in code with ephemeral IDs and excess entries are logged",
      len(capped_other[2]) == 10 and capped_other[2][-1]["id"] == "other:9"
      and "Eyelid twitching 10" in capped_log)
catalog_mismatch, mismatch_log = extracted_other({"present": [{"id": "fever", "evidence": "unmatched quote"}]})
check("catalog quote mismatches are counted but existing findings and quotes are retained",
      catalog_mismatch[:2] == ({"fever": True}, {"fever": "unmatched quote"})
      and "catalog evidence quote mismatches: 1" in mismatch_log)
check("empty extended extraction still abstains offline", extract.extract("", include_other=True) == ({}, {}, []))
check("new extraction schema and prompt participate in the cache version",
      extract._PROMPT_VERSION == llm.content_hash(extract.SYSTEM_PROMPT, json.dumps(extract._TOOL, sort_keys=True),
          json.dumps(vocabulary.catalog_for_prompt(), sort_keys=True, ensure_ascii=False))
      and extract._PROMPT_VERSION != llm.content_hash(extract.SYSTEM_PROMPT,
          json.dumps({"without": "other"}, sort_keys=True),
          json.dumps(vocabulary.catalog_for_prompt(), sort_keys=True, ensure_ascii=False)))


# ===========================================================================
section("supplemental proposal evidence stays outside catalog state")

other_evidence = [{"id": "other:0", "text": "Eyelid twitches 5 times", "evidence": "PRIVATE_QUOTE 95% probability"}]
other_candidate = {"condition": "Eyelid myokymia", "reasoning": "Eyelid twitching persists.",
                   "supported_by": ["other:0", "other:99"], "opposed_by": ["other:0"],
                   "unresolved": ["other:0", "calf_swelling_unilateral"]}


def proposed_other(items, candidate=other_candidate):
    keys = []

    def invoke(path, produce):
        keys.append(str(path))
        return produce()

    with patch.object(llm, "cached", side_effect=invoke), \
            patch.object(llm, "call_tool", return_value={"candidates": [candidate]}) as call, \
            patch("sys.stdout", io.StringIO()):
        result = propose.propose({}, items)
    return result, call, keys


other_candidates, other_call, other_keys = proposed_other(other_evidence)
check("supplemental-only evidence reaches proposal as a present text list without source quotes",
      other_call.call_count == 1 and "ALSO NOTED, NOT IN VOCABULARY" in other_call.call_args.args[1]
      and "other:0  (Eyelid twitches 5 times)" in other_call.call_args.args[1]
      and "PRIVATE_QUOTE" not in other_call.call_args.args[1]
      and "95% probability" not in other_call.call_args.args[1])
check("proposal keeps existing supplemental support and drops nonexistent supplemental references",
      len(other_candidates) == 1 and other_candidates[0]["supported_by"] == ["other:0"])
check("supplemental ids cannot become opposed or unknown proposal findings",
      other_candidates[0]["opposed_by"] == []
      and other_candidates[0]["unresolved"] == ["calf_swelling_unilateral"])
other_triage = eng.assess(other_candidates, {}, other_findings=other_evidence)
check("supplemental-only candidates do not abstain and unlisted conditions stay unrated",
      other_triage["abstaining"] is False and len(other_triage["unrated"]) == 1
      and other_triage["unrated"][0]["severity_band"] == "unrated")
check("supplemental support chips use clinical text instead of the ephemeral id",
      other_triage["unrated"][0]["supported_by"] == [{"finding": "other:0", "label": "Eyelid twitches 5 times"}])
check("supplemental evidence never enters catalog state or unknown-finding checks",
      other_triage["findings_present"] == other_triage["findings_absent"] == other_triage["doesnt_fit"] == []
      and "other:0" not in vocabulary.FINDINGS)
check("supplemental ids never supply the next question or its asked tracking",
      other_triage["next_question"]["finding"] == "calf_swelling_unilateral"
      and eng.assess(other_candidates, {}, asked=["calf_swelling_unilateral"], other_findings=other_evidence)["next_question"] is None)
rogue_other = dict(other_candidate, unresolved=["other:0"], opposed_by=["other:0"], supported_by=["other:99"])
rogue_payload = eng.assess([rogue_other], {}, other_findings=other_evidence)
check("triage defensively drops invalid supplemental support and never asks about supplemental evidence",
      rogue_payload["next_question"] is None and rogue_payload["unrated"][0]["supported_by"] == []
      and rogue_payload["unrated"][0]["opposed_by"] == [])
other_pe = dict(other_candidates[0], condition="Pulmonary embolism", unresolved=[])
unconfirmed_other = eng.assess([other_pe], {}, other_findings=other_evidence)
confirmed_other = eng.assess([other_pe], {}, confirmed_exclusions=["Acute pulmonary embolism"], other_findings=other_evidence)
check("supplemental evidence cannot exclude a condition without clinician confirmation",
      len(unconfirmed_other["cant_miss"]) == 1 and unconfirmed_other["ruled_out"] == []
      and unconfirmed_other["cant_miss"][0]["exclusion_options"] == eng.exclusion_options("Pulmonary embolism")
      and not any("other:" in json.dumps(option) for option in unconfirmed_other["cant_miss"][0]["exclusion_options"])
      and confirmed_other["cant_miss"] == [] and len(confirmed_other["ruled_out"]) == 1)
expanded_candidates, _, _ = proposed_other(other_evidence, dict(other_candidate, reasoning="other:0 was noted; other:99 is unavailable."))
check("supplemental citations in generated prose expand to text and never leak internal ids",
      "other:" not in expanded_candidates[0]["reasoning"]
      and "Eyelid twitches 5 times" in expanded_candidates[0]["reasoning"])
_, _, changed_other_keys = proposed_other([dict(other_evidence[0], text="Persistent eyelid twitching")])
_, _, changed_quote_keys = proposed_other([dict(other_evidence[0], evidence="DIFFERENT PRIVATE QUOTE")])
check("proposal caches include supplemental ids and text but never source quotes",
      other_keys != changed_other_keys and other_keys == changed_quote_keys)
check("proposal schema and prompt invalidate its version alongside finding labels",
      propose._PROMPT_VERSION == llm.content_hash(propose.SYSTEM_PROMPT, json.dumps(propose._TOOL, sort_keys=True),
          json.dumps({fid: meta["label"] for fid, meta in vocabulary.FINDINGS.items()}, sort_keys=True, ensure_ascii=False)))


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
section("documents can be selected by dropping files or using the picker")
class DropAreaMarkup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.parents = [], {}

    def handle_starttag(self, tag, attrs):
        identifier = dict(attrs).get("id")
        if identifier:
            self.parents[identifier] = tuple(item[1] for item in self.stack)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append((tag, identifier))

    def handle_endtag(self, tag):
        while self.stack:
            if self.stack.pop()[0] == tag:
                break


drop_markup = DropAreaMarkup()
drop_markup.feed((ROOT / "web" / "index.html").read_text(encoding="utf-8"))
check("the drop target contains the records heading, document list and file controls",
      all("doc-drop" in drop_markup.parents.get(identifier, ())
          for identifier in ("docn", "docs", "doc-files", "doc-send", "doc-selected", "doc-result")))
try:
    browser_result = browser_checks(r"""
const drop=document.getElementById('doc-drop'), picker=document.getElementById('doc-files');
const image={name:'<b>scan</b>.png',type:'image/png'}, pdf={name:'report.PDF',type:''};
const unsupported={name:'notes.exe',type:'application/octet-stream'};
let prevented=0, fetches=0;
function event(files=[]){return {target:drop,clientX:150,clientY:250,preventDefault(){prevented++;},dataTransfer:{files,types:['Files'],dropEffect:''}};}
drop.ondragenter(event());drop.ondragenter(event());drop.ondragleave({...event(),relatedTarget:drop});
results.nested_highlight=drop.classList.contains('dragging');
drop.ondragleave(event());
results.highlight_clears=!drop.classList.contains('dragging');
const over=event();drop.ondragover(over);
drop.ondrop(event([image,pdf]));
results.multiple_drop=run('docFiles.length===2') && !elements.get('doc-send').disabled
  && document.getElementById('doc-selected').textContent.includes('<b>scan</b>.png')
  && document.getElementById('doc-selected').innerHTML==='';
results.prevent_navigation=prevented===6 && over.dataTransfer.dropEffect==='copy'
  && !drop.classList.contains('dragging');
const additional={name:'another-report.pdf',type:'application/pdf',size:123,lastModified:42};
drop.ondrop(event([additional]));
results.repeated_drops_add=run('docFiles.length===3')
  && document.getElementById('doc-selected').textContent.includes('another-report.pdf')
  && document.getElementById('doc-selected').textContent.includes('report.PDF');
drop.ondrop(event([additional,image]));
results.duplicate_drops_skip=run('docFiles.length===3');
run('clearDocSelection()');
drop.ondrop(event([image,unsupported]));
results.mixed_types=run('docFiles.length===1 && docFiles[0].name.includes("scan")')
  && document.getElementById('doc-selected').textContent.includes('Unsupported files skipped: notes.exe');
drop.ondrop(event([unsupported]));
results.unsupported_preserves=run('docFiles.length===1') && !elements.get('doc-send').disabled;
run('clearDocSelection()');drop.ondrop(event([unsupported]));
results.unsupported_disabled=run('docFiles.length===0') && elements.get('doc-send').disabled;
picker.files=[pdf];picker.onchange();
drop.ondrop(event());
results.picker_and_empty_drop=run(`docFiles.length===1 && docFiles[0].name==='report.PDF'`)
  && document.getElementById('doc-selected').textContent.includes('report.PDF');

let finishUpload, sentFiles=[];
context.FormData=class {append(key,file,name){sentFiles.push({key,name});}};
context.fetch=()=>{fetches++;return new Promise(resolve=>{finishUpload=resolve;});};
const upload=run('docSend()');
results.upload_busy=picker.disabled && drop.classList.contains('busy');
drop.ondrop(event([image]));await run('docSend()');
results.pending_upload_protected=fetches===1 && sentFiles.length===1 && sentFiles[0].name==='report.PDF'
  && run(`docFiles.length===1 && docFiles[0].name==='report.PDF'`);
finishUpload({ok:true,json:async()=>({scanned_documents:[]})});await upload;
results.upload_cleanup=!picker.disabled && !drop.classList.contains('busy')
  && run('docFiles.length===0') && elements.get('doc-send').disabled
  && document.getElementById('doc-selected').textContent==='';
picker.files=[image];picker.onchange();drop.ondragenter(event());run('reset()');
results.reset_cleanup=run('docFiles.length===0') && picker.value===''
  && !drop.classList.contains('dragging') && elements.get('doc-send').disabled
  && document.getElementById('doc-selected').textContent==='';
""")
    browser_error = ""
except Exception as exc:
    browser_result, browser_error = {}, str(exc)
for key, description in [
    ("nested_highlight", "dragging over children keeps the drop area highlighted"),
    ("highlight_clears", "leaving the drop area clears its highlight"),
    ("multiple_drop", "dropping multiple images and PDFs selects them and renders filenames as text"),
    ("prevent_navigation", "dropping files prevents browser navigation and advertises copying"),
    ("repeated_drops_add", "successive drops accumulate files before scanning"),
    ("duplicate_drops_skip", "dropping the same file again does not duplicate its upload"),
    ("mixed_types", "supported dropped files survive alongside a visible unsupported-file message"),
    ("unsupported_preserves", "an unsupported drop does not discard already selected files"),
    ("unsupported_disabled", "unsupported files cannot enable scanning"),
    ("picker_and_empty_drop", "the file picker shares selection and text-only drops do not erase it"),
    ("upload_busy", "uploading temporarily disables the file picker and marks the drop area busy"),
    ("pending_upload_protected", "drops and duplicate scans cannot replace or resend an active upload"),
    ("upload_cleanup", "finishing an upload clears the selection and restores upload controls"),
    ("reset_cleanup", "reset clears pending files and drag feedback"),
]:
    check(description, browser_result.get(key, False), browser_error)


# ===========================================================================
section("file drops are intercepted before nested controls and cannot navigate")
try:
    browser_result = browser_checks(r"""
const drop=document.getElementById('doc-drop');
const child={id:'nested-control'}, outside={id:'outside'};
drop.contains=target=>target===drop || target===child;
const image={name:'scan.png',type:'image/png'};
function fileEvent(target,files=[],types=['Files']){
  return {target,clientX:150,clientY:250,defaultPrevented:false,preventDefault(){this.defaultPrevented=true;},
    dataTransfer:{files,types,dropEffect:''}};
}
function capture(type,event){
  for(const listener of documentListeners.get(type)||[]){
    if(listener.capture===true)listener.handler(event);
  }
}
const over=fileEvent(child);
capture('dragover',over);
results.capture_over=over.defaultPrevented && over.dataTransfer.dropEffect==='copy';
const inside=fileEvent(child,[image]);
capture('drop',inside);
results.capture_drop=inside.defaultPrevented && run(`docFiles.length===1 && docFiles[0].name==='scan.png'`)
  && !elements.get('doc-send').disabled;
const status=document.getElementById('doc-result');
status.textContent='Existing scan status';status.style.display='none';
const elsewhere=fileEvent(outside,[image]);
capture('dragover',elsewhere);capture('drop',elsewhere);
results.no_navigation_outside=elsewhere.defaultPrevented && elsewhere.dataTransfer.dropEffect==='none'
  && status.textContent==='Existing scan status' && status.style.display==='none';
let rejectedEverywhere=true;
for(const id of ['tx','dnote-input','cm','dx','qbody','mis','live-rec','patientbar','setupForm','doc-modal-body']){
  const target=document.getElementById(id);
  const event=fileEvent(target,[{name:id+'.pdf',type:'application/pdf'}]);
  capture('dragover',event);capture('drop',event);
  rejectedEverywhere=rejectedEverywhere && event.defaultPrevented
    && event.dataTransfer.dropEffect==='none'
    && run(`docFiles.length===1 && docFiles[0].name==='scan.png'`)
    && status.textContent==='Existing scan status';
}
results.strict_panel_boundary=rejectedEverywhere;
const directOutside=fileEvent(outside,[{name:'outside.pdf',type:'application/pdf'}]);
context.outsideDrop=directOutside;run('acceptDocDrop(outsideDrop)');
results.shared_handler_rejects_outside=run(`docFiles.length===1 && docFiles[0].name==='scan.png'`);
drop.classList.add('dragging');
const outsideHover=fileEvent(outside);
outsideHover.clientY=600;capture('dragover',outsideHover);
results.outside_hover_clears=!drop.classList.contains('dragging') && outsideHover.dataTransfer.dropEffect==='none';
let geometryRejected=true;
for(const [x,y] of [[99,250],[400,250],[150,199],[150,500]]){
  const stale=fileEvent(child,[{name:'outside-bounds.pdf',type:'application/pdf'}]);
  stale.clientX=x;stale.clientY=y;
  drop.classList.add('dragging');
  capture('dragover',stale);capture('drop',stale);
  geometryRejected=geometryRejected && stale.dataTransfer.dropEffect==='none'
    && !drop.classList.contains('dragging')
    && run(`docFiles.length===1 && docFiles[0].name==='scan.png'`);
}
results.geometry_rejected=geometryRejected;
capture('dragover',fileEvent(child));
results.reentry_highlights=drop.classList.contains('dragging');
drop.classList.add('dragging');
capture('dragend',fileEvent(child));
results.cancel_clears=!drop.classList.contains('dragging');
const text=fileEvent(child,[],['text/plain']);
context.testImage=image;run('selectDocFiles([testImage])');
capture('dragover',text);capture('drop',text);
results.text_untouched=!text.defaultPrevented && run('docFiles.length===1');
run('docScanInFlight=true');
const busy=fileEvent(child,[{name:'replacement.pdf',type:'application/pdf'}]);
capture('dragover',busy);capture('drop',busy);
results.busy_protected=busy.defaultPrevented && busy.dataTransfer.dropEffect==='none'
  && run(`docFiles.length===1 && docFiles[0].name==='scan.png'`);
""")
    browser_error = ""
except Exception as exc:
    browser_result, browser_error = {}, str(exc)
for key, description in [
    ("capture_over", "file dragover is accepted before nested control handlers"),
    ("capture_drop", "a file drop selects documents even if a child stops bubbling"),
    ("no_navigation_outside", "outside file drops are ignored without navigation or UI changes"),
    ("strict_panel_boundary", "every other consultation section rejects document drops"),
    ("shared_handler_rejects_outside", "the shared selection handler itself rejects outside drops"),
    ("outside_hover_clears", "moving over another section immediately clears the records highlight"),
    ("geometry_rejected", "all four visible panel boundaries reject outside pointer coordinates"),
    ("reentry_highlights", "returning inside the visible records bounds restores the highlight"),
    ("cancel_clears", "ending a drag clears the drop highlight"),
    ("text_untouched", "the file navigation guard leaves text drags alone"),
    ("busy_protected", "early file-drop handling cannot replace an active upload"),
]:
    check(description, browser_result.get(key, False), browser_error)


# ===========================================================================
section("scanned documents can be set aside and restored")
try:
    browser_result = browser_checks(r"""
let rows=[];
const docs=document.getElementById('docs');
Object.defineProperty(docs,'innerHTML',{set(){rows=[];},get(){return '';}});
docs.appendChild=row=>rows.push(row);
document.createElement=()=>{
  const row=element(), thumb=element(), toggle=element();
  row.querySelector=selector=>selector==='.thumb' ? thumb
    : selector==='button[data-doc]' && /<button[^>]*data-doc=/.test(row.innerHTML) ? toggle : null;
  return row;
};
let scheduled=0;
context.setTimeout=()=>{scheduled++;return scheduled;};
run(`liveTranscriptExtra='recorded consultation';scannedDocs=[
  {name:'clinical.txt',text:'first document text',chars:19,clinical:true,kind:'clinical record',use:true},
  {name:'other.txt',text:'other document text',chars:19,clinical:true,kind:'clinical record',use:true},
  {name:'rejected.txt',text:'retained rejected text',chars:22,clinical:false,kind:'screenshot',use:false}
];renderScannedDocs();`);
const original=run('JSON.stringify(scannedDocs[0])');
const aside=rows[0].querySelector('button[data-doc]');
results.clinical_control=!!aside && rows[0].innerHTML.includes('Set aside');
if(aside){
  aside.onclick();
  results.excludes_selected=run(`!scannedDocs[0].use && docText()==='other document text'
    && liveTranscriptExtra==='recorded consultation'`) && elements.get('docn').textContent==='1 of 3 used';
  results.schedules_analysis=scheduled===1;
  rows[0].querySelector('.thumb').onclick();
  results.retains_text=elements.get('doc-modal-body').textContent==='first document text';
  const restore=rows[0].querySelector('button[data-doc]');
  if(restore)restore.onclick();
  results.restores_document=run('JSON.stringify(scannedDocs[0])')===original
    && run(`docText()==='first document text\\n\\nother document text'`) && scheduled===2;
}
const rejected=rows[2].querySelector('button[data-doc]');
if(rejected)rejected.onclick();
results.restores_rejected=run(`scannedDocs[2].use && docText().includes('retained rejected text')`);
""")
    browser_error = ""
except Exception as exc:
    browser_result, browser_error = {}, str(exc)
for key, description in [
    ("clinical_control", "included clinical documents expose a Set aside control"),
    ("excludes_selected", "setting aside removes only that document from analysis"),
    ("schedules_analysis", "setting aside schedules analysis with the updated evidence"),
    ("retains_text", "a set-aside document's extracted text remains readable"),
    ("restores_document", "restoring a clinical document preserves its text and screening metadata"),
    ("restores_rejected", "a screened-out document can still be included manually"),
]:
    check(description, browser_result.get(key, False), browser_error)


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
section("independent chat models, V2 transport and cache invalidation")
with patch.dict(os.environ, {}, clear=True):
    check("default chat models are Gemma screening and separate GLM clinical calls",
          [llm.config(task)["model"] for task in ("screen", "extract", "propose")]
          == ["gemma4", "glm5.3", "glm5.3"])
    original_profiles = {task: llm.config_key(task) for task in ("screen", "extract", "propose")}
    with patch.dict(os.environ, {"SARVAM_EXTRACT_MODEL": "deepseekv4.1-flash"}):
        check("changing extraction model invalidates only extraction's profile",
              llm.config_key("extract") != original_profiles["extract"]
              and all(llm.config_key(task) == original_profiles[task] for task in ("screen", "propose")))
    with patch.dict(os.environ, {"SARVAM_CHAT_MODEL": "sarvam-105b", "SARVAM_SCREEN_MODEL": "gemma4"}):
        check("legacy chat override still works and task settings take precedence",
              llm.config("extract")["api_version"] == "v1"
              and llm.config("propose")["model"] == "sarvam-105b"
              and llm.config("screen")["model"] == "gemma4")
    with patch.dict(os.environ, {"SARVAM_EXTRACT_MODEL": "unknown-model"}):
        try:
            llm.config("extract")
            invalid_model_rejected = False
        except ValueError:
            invalid_model_rejected = True
        check("unknown chat model names fail before a paid request", invalid_model_rejected)


def offline_v2(replies, *, task="extract"):
    requests = []

    def respond(request):
        requests.append((str(request.url), json.loads(request.content)))
        status, body = replies[min(len(requests) - 1, len(replies) - 1)]
        return httpx.Response(status, json=body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport, \
            patch.object(llm, "http_client", return_value=transport), \
            patch.object(llm, "client", side_effect=AssertionError("V2 must not use the V1 SDK")), \
            patch.object(llm.time, "sleep"), \
            patch.dict(os.environ, {"SARVAM_API_KEY": "offline-test-key"}, clear=True):
        try:
            return llm.call_tool("offline system", "offline input", extract._SCREEN_TOOL, task=task), requests, None
        except Exception as exc:
            return None, requests, exc


v2_success = {"choices": [{"message": {"tool_calls": [{"function": {
    "name": extract._SCREEN_TOOL["function"]["name"], "arguments": json.dumps(screen_arguments),
}}]}}]}
result, requests, error = offline_v2([(200, v2_success)])
check("V2 returns the requested structured tool object", result == screen_arguments and error is None)
body = requests[0][1]
check("GLM uses the V2 endpoint with an explicit reasoning budget",
      requests[0][0] == "https://api.sarvam.ai/v2/chat/completions"
      and body["model"] == "glm5.3" and body["reasoning_effort"] == "high"
      and body["max_tokens"] == 16384 and body["temperature"] == 0 and body["seed"] == 0
      and body["tool_choice"]["function"]["name"] == extract._SCREEN_TOOL["function"]["name"])
result, requests, error = offline_v2([(200, v2_success)], task="screen")
check("Gemma screening keeps its separate short nonreasoning request",
      result == screen_arguments and requests[0][1]["model"] == "gemma4"
      and requests[0][1]["max_tokens"] == 256 and requests[0][1]["reasoning_effort"] is None)
for label, bad in (
    ("empty choices", {"choices": []}),
    ("malformed tool entry", {"choices": [{"message": {"tool_calls": [None]}}]}),
    ("nonobject arguments", {"choices": [{"message": {"tool_calls": [{"function": {
        "name": extract._SCREEN_TOOL["function"]["name"], "arguments": "[]"}}]}}]}),
    ("wrong tool name", {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "wrong_tool", "arguments": "{}"}}]}}]}),
    ("invalid JSON", {"choices": [{"message": {"tool_calls": [{"function": {
        "name": extract._SCREEN_TOOL["function"]["name"], "arguments": "broken JSON"}}]}}]}),
):
    result, requests, error = offline_v2([(200, bad), (200, v2_success)])
    check(f"V2 {label} retries without becoming empty findings",
          result == screen_arguments and error is None and len(requests) == 2
          and [request[1]["seed"] for request in requests] == [0, 1])
result, requests, error = offline_v2([(429, {"error": "rate limited"}), (200, v2_success)])
check("V2 throttling retries with a fresh seed", result == screen_arguments and len(requests) == 2)
result, requests, error = offline_v2([(403, {"error": "model not enabled"})])
check("V2 access failures are surfaced without silently changing models",
      result is None and isinstance(error, httpx.HTTPStatusError) and len(requests) == 1)

# Exercise each real caller's cache key, not just the shared configuration helper.
for task, invoke, cached_value in [
    ("screen", lambda: extract.screen_document("offline document"), screen_arguments),
    ("extract", lambda: extract.extract("offline transcript"), {"present": [], "absent": []}),
    ("propose", lambda: propose.propose({"headache": True}), {"candidates": []}),
]:
    paths = []
    def record_cache(path, produce):
        paths.append(path)
        return cached_value
    with patch.dict(os.environ, {}, clear=True), patch.object(llm, "cached", side_effect=record_cache):
        invoke()
        invoke()
        with patch.dict(os.environ, {f"SARVAM_{task.upper()}_MODEL": "sarvam-105b"}):
            invoke()
    check(f"{task} caches unchanged inputs but invalidates on model changes",
          paths[0] == paths[1] and paths[0] != paths[2])


# ===========================================================================
section("capture model selection and model-aware caches stay offline")
with patch.dict(os.environ, {}, clear=True):
    check("Hindi and English use their own documented v4 voices",
          capture.speech_config("hi-IN") == {"model": "bulbul:v4-flash", "speaker": "aparna_hi_customer"}
          and capture.speech_config("en-IN") == {"model": "bulbul:v4-flash", "speaker": "aparna_en_companion"})
    check("remaining supported languages retain v3 instead of an English persona",
          all(capture.speech_config(code) == {"model": "bulbul:v3", "speaker": "shubh"}
              for code in ("bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN")))

with tempfile.TemporaryDirectory(dir=ROOT / "cache", prefix="selftest-capture-") as capture_test_dir:
    fake = Mock()
    fake.text_to_speech.convert.return_value = SimpleNamespace(audios=[base64.b64encode(b"offline wav").decode()])
    fake.text.translate.return_value = SimpleNamespace(translated_text="offline translated question")
    job = fake.speech_to_text_job.create_job.return_value
    job.is_failed.return_value = False
    def write_transcript(*, output_dir):
        (Path(output_dir) / "question.wav.json").write_text(json.dumps({"transcript": "offline question"}), encoding="utf-8")
    job.download_outputs.side_effect = write_transcript
    audio_path = Path(capture_test_dir) / "question.wav"
    audio_path.write_bytes(b"offline audio")
    with patch.object(capture, "CACHE_DIR", Path(capture_test_dir)), \
            patch.object(capture, "client", return_value=fake), patch.dict(os.environ, {}, clear=True):
        capture.speak("offline question", "hi-IN")
        capture.speak("offline question", "hi-IN")
        check("v4 WAV generation is cached with explicit codec and sample rate",
              fake.text_to_speech.convert.call_count == 1
              and fake.text_to_speech.convert.call_args.kwargs["model"] == "bulbul:v4-flash"
              and fake.text_to_speech.convert.call_args.kwargs["output_audio_codec"] == "wav"
              and fake.text_to_speech.convert.call_args.kwargs["speech_sample_rate"] == 24000)
        with patch.dict(os.environ, {"SARVAM_TTS_MODEL": "bulbul:v3"}):
            capture.speak("offline question", "hi-IN")
        check("switching TTS models cannot reuse old audio",
              fake.text_to_speech.convert.call_count == 2
              and fake.text_to_speech.convert.call_args.kwargs["speaker"] == "shubh")
        capture.translate("offline question", "en-IN", "hi-IN")
        capture.translate("offline question", "en-IN", "hi-IN")
        check("translation pins sarvam-translate:v1 and caches repeated questions",
              fake.text.translate.call_count == 1
              and fake.text.translate.call_args.kwargs["model"] == "sarvam-translate:v1")
        with patch.dict(os.environ, {"SARVAM_TRANSLATION_MODEL": "mayura:v1"}):
            capture.translate("offline question", "en-IN", "hi-IN")
        check("translation model changes invalidate translated text", fake.text.translate.call_count == 2)
        capture.transcribe(audio_path, "en-IN")
        capture.transcribe(audio_path, "en-IN")
        check("Saaras v4 batch STT retains diarization and caches repeated audio",
              fake.speech_to_text_job.create_job.call_count == 1
              and fake.speech_to_text_job.create_job.call_args.kwargs["model"] == "saaras:v4"
              and fake.speech_to_text_job.create_job.call_args.kwargs["with_diarization"] is True)
        with patch.dict(os.environ, {"SARVAM_STT_MODEL": "saaras:v3"}):
            capture.transcribe(audio_path, "en-IN")
        check("STT model changes invalidate old transcripts", fake.speech_to_text_job.create_job.call_count == 2)


# ===========================================================================
section("unrated logging contains no consultation evidence")
import main

with tempfile.TemporaryDirectory(dir=ROOT / "cache", prefix="selftest-unrated-") as log_test_dir:
    log_path = Path(log_test_dir) / "cache" / "unrated_log.jsonl"
    request = main.AnalyzeRequest(transcript="PRIVATE_TRANSCRIPT", doc_text="PRIVATE_DOCUMENT")
    with patch.object(main, "REPO_ROOT", Path(log_test_dir)), \
            patch.object(main.extract, "extract", return_value=(FINDINGS, {"breathlessness": "PRIVATE_QUOTE"}, [])), \
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
section("supplemental evidence reaches the API and escaped clinician panels")


async def exercise_supplemental_pipeline():
    source = "My eyelid twitches <b>x</b> 5 times; SOURCE_ONLY 95% probability."
    phrase = "Eyelid twitching <b>x</b> 5 times"
    proposed_inputs, extraction_keys, assessment_calls = [], [], []
    invalid_quote = False

    def response(system, user, tool, *, task):
        if task == "extract":
            return {"present": [], "absent": [], "other": [{
                "text": phrase if not invalid_quote else "Unquoted twitching",
                "evidence": source if not invalid_quote else "This was never said.",
            }]}
        if task != "propose":
            raise AssertionError("Unexpected model task in offline test")
        proposed_inputs.append(user)
        return {"candidates": [{
            "condition": "Eyelid myokymia",
            "reasoning": "other:0 was reported. A 95% chance of recovery.",
            "supported_by": ["other:0", "other:99"], "opposed_by": ["other:0"],
            "unresolved": ["other:0", "calf_swelling_unilateral"],
        }]}

    def cached(path, produce):
        extraction_keys.append(str(path))
        return produce()

    original_assess = eng.assess

    def assess(*args, **kwargs):
        assessment_calls.append((args, kwargs))
        return original_assess(*args, **kwargs)

    logs = io.StringIO()
    with patch.object(llm, "cached", side_effect=cached), patch.object(llm, "call_tool", side_effect=response), \
            patch.object(main, "get_engine", return_value=eng), patch.object(eng, "assess", side_effect=assess), \
            patch.object(main, "_log_unrated"), patch("sys.stdout", logs):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://offline.test") as client:
            result = await client.post("/analyze", json={"doc_text": source, "asked": ["other:0", "calf_swelling_unilateral"]})
            invalid_quote = True
            rejected = await client.post("/analyze", json={"transcript": source})
    return result, rejected, proposed_inputs, assessment_calls, logs.getvalue(), source, phrase


api_other, api_rejected, api_proposal_inputs, api_assessments, api_other_log, api_source, api_phrase = asyncio.run(exercise_supplemental_pipeline())
api_payload = api_other.json()
check("analyze returns validated supplemental evidence with its verbatim numeric quote",
      api_other.status_code == 200
      and api_payload["other_findings"] == [{"id": "other:0", "text": api_phrase, "evidence": api_source}])
check("an other-only consultation reaches proposal and returns candidates without abstaining",
      len(api_proposal_inputs) == 1 and not api_payload["abstaining"] and bool(api_payload["unrated"])
      and api_payload["_findings"] == api_payload["_quotes"] == {}
      and "SOURCE_ONLY" not in api_proposal_inputs[0] and api_source not in api_proposal_inputs[0])
check("analyze passes only catalog ids into asked tracking and no supplemental exclusion state",
      api_assessments[0][1]["asked"] == ["calf_swelling_unilateral"]
      and api_assessments[0][1]["confirmed_exclusions"] == []
      and api_assessments[0][0][1] == {} and api_payload["next_question"] is None
      and api_payload["findings_present"] == api_payload["findings_absent"] == [])
check("an unquoted supplemental claim is omitted from the API and logged without proposing from it",
      api_rejected.status_code == 200 and api_rejected.json()["other_findings"] == []
      and api_rejected.json()["abstaining"] and "Unquoted twitching" in api_other_log)
generated_other = [api_payload["other_findings"][0]["text"], api_payload["unrated"][0]["reasoning"],
                   api_payload["unrated"][0]["supported_by"][0]["label"]]
check("supplemental generated text contains no percentages or probability wording while quotes remain untouched",
      all(not propose._UNSUPPORTED_REASONING.search(value) for value in generated_other)
      and "95% probability" in api_payload["other_findings"][0]["evidence"]
      and "5" in api_payload["other_findings"][0]["text"])

# Load definitions only: never invoke run_cases.main() or any live case.
case_helpers = runpy.run_path(str(ROOT / "run_cases.py"), run_name="offline_case_helpers")
number_issues = case_helpers["number_issues"]
check("manual case checks ignore internal ids and verbatim evidence but permit ordinary clinical digits",
      not number_issues(api_payload)
      and not number_issues({"id": "other:95", "finding": "other:95", "evidence": "95% probability", "text": "Twitches 5 times"}))
for field, value in [("label", "95%"), ("reasoning", "A chance of recovery"),
                     ("note", "High confidence"), ("text", "High probability"),
                     ("note", "Risk score 95"), ("score", 5), ("weight", 0.5)]:
    check(f"manual case checks still inspect generated {field}: {value}", bool(number_issues({field: value})))

try:
    browser_result = browser_checks(r"""
const payload = PAYLOAD;
run('committed=false;hasLiveInput=true;');context.payload=payload;
run('dxState={stage:payload};cmState={stage:payload};paintDX();');
results.hidden_without_note=elements.get('other-card').classList.contains('hide') && elements.get('other-findings').innerHTML==='';
run('committed=true;hasLiveInput=false;paintDX();');
results.hidden_without_input=elements.get('other-card').classList.contains('hide') && elements.get('other-findings').innerHTML==='';
run('committed=true;hasLiveInput=true;paintDX();paintCM();paintQB();paintMIS();');
const html=elements.get('other-findings').innerHTML;
results.visible_and_escaped=!elements.get('other-card').classList.contains('hide')
  && (html.match(/&lt;b&gt;x&lt;\/b&gt;/g)||[]).length===2 && !html.includes('<b>x</b>')
  && html.includes('5 times') && html.includes('95% probability');
results.support_label=elements.get('dx').innerHTML.includes('Eyelid twitching &lt;b&gt;x&lt;/b&gt; 5 times');
results.no_displayed_ids=[...elements.values()].every(el=>!el.innerHTML.includes('other:') && !el.textContent.includes('other:'));
run('setAnalysisError("offline failure")');
results.stale=elements.get('other-findings').classList.contains('analysis-stale');
run(`dxState.stage.next_question={finding:'other:0',ask:'other:0'};paintQB();`);
results.no_supplemental_question=!elements.get('qbody').innerHTML.includes('other:') && run('askedFindings.length===0');
let sent;
context.fetch=async(url,options)=>{sent=JSON.parse(options.body);return {ok:true,json:async()=>payload};};
run(`askedFindings=['other:0','calf_swelling_unilateral'];liveTranscriptExtra='My eyelid twitches.';`);
await run('runAnalyze()');
results.no_supplemental_asked=JSON.stringify(sent.asked)===JSON.stringify(['calf_swelling_unilateral']);
run("scannedDocs=[];liveTranscriptExtra='';");await run('runAnalyze()');
results.cleared_without_evidence=elements.get('other-card').classList.contains('hide') && elements.get('other-findings').innerHTML==='';
run('dxState={stage:payload};paintDX();reset();');
results.reset_clears=elements.get('other-card').classList.contains('hide') && elements.get('other-findings').innerHTML==='';
""".replace("PAYLOAD", json.dumps(api_payload)))
    browser_error = ""
except Exception as exc:
    browser_result, browser_error = {}, str(exc)
check("the supplemental section has an explicit outside-the-standard-list label",
      "Also noted, outside the standard list" in (ROOT / "web/index.html").read_text(encoding="utf-8"))
for key, description in [
    ("hidden_without_note", "supplemental evidence stays hidden until the clinician note is committed"),
    ("hidden_without_input", "supplemental evidence stays hidden without real input"),
    ("visible_and_escaped", "supplemental text and verbatim quotes both escape HTML while retaining clinical numbers"),
    ("support_label", "candidate support rows show the supplemental text"),
    ("no_displayed_ids", "other: never appears anywhere in the rendered page"),
    ("stale", "failed analysis marks supplemental evidence as stale with the other panels"),
    ("no_supplemental_question", "even a malformed supplemental question cannot render or enter asked tracking"),
    ("no_supplemental_asked", "analysis requests omit supplemental ids from asked tracking"),
    ("cleared_without_evidence", "removing all evidence clears the supplemental section"),
    ("reset_clears", "reset clears supplemental evidence"),
]:
    check(description, browser_result.get(key, False), browser_error)


# ===========================================================================
section("document scan failures stay local to each file")


async def exercise_document_batches():
    loop = asyncio.get_running_loop()
    failure_seen = asyncio.Event()
    release = threading.Event()
    started = threading.Barrier(3, timeout=5)
    finished = threading.Event()
    lock = threading.Lock()
    paths, finished_count = [], 0

    def read_one(path):
        nonlocal finished_count
        path = Path(path)
        marker = path.read_text(encoding="utf-8")
        with lock:
            paths.append(path)
        try:
            started.wait()
            if marker == "broken":
                loop.call_soon_threadsafe(failure_seen.set)
                raise RuntimeError("offline OCR failure")
            if not release.wait(timeout=5):
                raise RuntimeError("offline scan was not released")
            return {"text": path.read_text(encoding="utf-8") + " output"}
        finally:
            with lock:
                finished_count += 1
                if finished_count == 3:
                    finished.set()

    transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
    with patch.object(main.capture, "read_image", side_effect=read_one), \
            patch.object(main.extract, "screen_document", return_value={"clinical": True, "kind": "offline record"}), \
            patch("sys.stderr", io.StringIO()):
        async with httpx.AsyncClient(transport=transport, base_url="http://offline") as client:
            request = asyncio.create_task(client.post("/capture", files=[
                ("documents", ("first.png", b"first", "image/png")),
                ("documents", ("broken.png", b"broken", "image/png")),
                ("documents", ("last.png", b"last", "image/png")),
            ]))
            try:
                await asyncio.wait_for(failure_seen.wait(), timeout=5)
                done, _ = await asyncio.wait({request}, timeout=0.05)
                waits_for_all = not done
                files_live = len(paths) == 3 and all(path.exists() for path in paths)
            finally:
                release.set()
            response = await asyncio.wait_for(request, timeout=5)
            await asyncio.to_thread(finished.wait, 5)
    try:
        mixed = response.json()
    except ValueError:
        mixed = {}
    check("a mixed document batch returns HTTP 200", response.status_code == 200, response.text[:200])
    rows = mixed.get("scanned_documents", [])
    check("all document results retain their upload order",
          [row["name"] for row in rows] == ["first.png", "broken.png", "last.png"])
    check("successful documents survive a sibling OCR failure",
          len(rows) == 3 and rows[0].get("text") == "first output"
          and rows[2].get("text") == "last output" and rows[0].get("clinical") is True)
    check("an unreadable document is an error entry with no usable text",
          len(rows) == 3 and rows[1].get("status") == "error" and rows[1].get("clinical") is False
          and rows[1].get("chars") == 0 and rows[1].get("text") == ""
          and 0 < len(rows[1].get("error", "")) <= 200)
    check("a failed scan does not finish the request while sibling scans are pending", waits_for_all)
    check("all temporary files remain available until the other scans finish", files_live)
    check("temporary files are removed after the complete batch", bool(paths) and all(not path.exists() for path in paths))

    with patch.object(main.capture, "read_image", side_effect=TimeoutError("offline timeout")), \
            patch("sys.stderr", io.StringIO()):
        async with httpx.AsyncClient(transport=transport, base_url="http://offline") as client:
            response = await client.post("/capture", files=[
                ("documents", ("one.png", b"one", "image/png")),
                ("documents", ("two.png", b"two", "image/png")),
            ])
            all_failed = response.json()
            malformed = await client.post("/capture", files=[("documents", (None, "not a file"))])
            wrong_type = await client.post("/capture", json={"documents": []})
    errors = all_failed.get("scanned_documents", [])
    check("an entirely unreadable batch still returns HTTP 200 with every error",
          response.status_code == 200 and len(errors) == 2
          and all(row.get("status") == "error" and row.get("clinical") is False for row in errors))
    check("a malformed multipart document entry remains a request-level error", malformed.status_code == 502)
    check("the existing non-multipart validation is unchanged", wrong_type.status_code == 400)

    cancellation_started = asyncio.Event()
    cancellation_release = threading.Event()
    cancellation_paths = []

    def read_until_released(path):
        with lock:
            cancellation_paths.append(Path(path))
            if len(cancellation_paths) == 2:
                loop.call_soon_threadsafe(cancellation_started.set)
        if not cancellation_release.wait(timeout=5):
            raise RuntimeError("offline cancellation scan was not released")
        return {"text": Path(path).read_text(encoding="utf-8")}

    with patch.object(main.capture, "read_image", side_effect=read_until_released), \
            patch.object(main.extract, "screen_document", return_value={"clinical": True, "kind": "offline record"}), \
            patch("sys.stderr", io.StringIO()):
        async with httpx.AsyncClient(transport=transport, base_url="http://offline") as client:
            request = asyncio.create_task(client.post("/capture", files=[
                ("documents", ("one.png", b"one", "image/png")),
                ("documents", ("two.png", b"two", "image/png")),
            ]))
            try:
                await asyncio.wait_for(cancellation_started.wait(), timeout=5)
                request.cancel()
                done, _ = await asyncio.wait({request}, timeout=0.05)
                cancel_waits = not done and all(path.exists() for path in cancellation_paths)
            finally:
                cancellation_release.set()
            cancelled = False
            try:
                await asyncio.wait_for(request, timeout=5)
            except asyncio.CancelledError:
                cancelled = True
    check("request cancellation waits for scans before removing their files", cancel_waits)
    check("request cancellation still propagates after scans finish", cancelled)
    check("cancelled requests clean up files after every worker finishes",
          len(cancellation_paths) == 2 and all(not path.exists() for path in cancellation_paths))
    return mixed, all_failed


mixed_documents, failed_documents = asyncio.run(exercise_document_batches())


# ===========================================================================
section("failed document rows are visible but never become analysis evidence")
try:
    browser_result = browser_checks(r"""
let rows=[];
const docs=document.getElementById('docs');
Object.defineProperty(docs,'innerHTML',{set(){rows=[];},get(){return '';}});
docs.appendChild=row=>rows.push(row);
document.createElement=()=>{
  const row=element(), thumb=element(), toggle=element();
  row.querySelector=selector=>selector==='.thumb' ? thumb
    : selector==='button[data-doc]' && /<button[^>]*data-doc=/.test(row.innerHTML) ? toggle : null;
  return row;
};
context.FormData=class {append(){}};
context.scanResponse=MIXED_DOCUMENTS;
const analysisBodies=[];
context.fetch=async(url,options)=>{
  if(url==='/capture')return {ok:true,json:async()=>context.scanResponse};
  if(url==='/analyze'){
    analysisBodies.push(JSON.parse(options.body));
    return {ok:true,json:async()=>({abstaining:true,message:'Offline test',cant_miss:[],
      differential:[],unrated:[],ruled_out:[],doesnt_fit:[],next_question:null})};
  }
  throw new Error('Unexpected offline request');
};
run(`reset();committed=true;docFiles=[{name:'first.png'},{name:'broken.png'},{name:'last.png'}];`);
await run('docSend()');
results.mixed_summary=document.getElementById('doc-result').textContent.includes('2 scanned, 1 could not be read')
  && document.getElementById('docn').textContent.includes('2 scanned, 1 could not be read');
results.error_visible=rows.length===3 && rows[1].innerHTML.toLowerCase().includes('could not be read')
  && rows[1].innerHTML.includes('broken.png')
  && rows[1].innerHTML.includes(context.scanResponse.scanned_documents[1].error)
  && !rows[1].querySelector('button[data-doc]');
results.failed_excluded=run(`scannedDocs.length===3 && scannedDocs[1].use===false
  && docText()==='first output\\n\\nlast output'`);
if(run('scannedDocs.length===3')){
  run(`scannedDocs[1].use=true;scannedDocs[1].text='DO_NOT_ANALYZE';`);
  await run('runAnalyze()');
  results.analysis_only_success=analysisBodies.length===1
    && analysisBodies[0].doc_text==='first output\n\nlast output';
  run(`scannedDocs[1].error='<img src=x onerror=bad()> could not be read';renderScannedDocs();`);
  results.escaped_error=rows[1].innerHTML.includes('&lt;img') && !rows[1].innerHTML.includes('<img');
}
context.scanResponse=FAILED_DOCUMENTS;
run(`reset();docFiles=[{name:'one.png'},{name:'two.png'}];`);await run('docSend()');
results.all_failed=run('scannedDocs.length===2 && scannedDocs.every(d=>!d.use) && docText()===\'\' && !hasLiveInput')
  && document.getElementById('doc-result').textContent.includes('0 scanned, 2 could not be read');
""".replace("MIXED_DOCUMENTS", json.dumps(mixed_documents)).replace("FAILED_DOCUMENTS", json.dumps(failed_documents)))
    browser_error = ""
except Exception as exc:
    browser_result, browser_error = {}, str(exc)
for key, description in [
    ("mixed_summary", "a mixed batch displays scanned and unreadable document counts"),
    ("error_visible", "an unreadable row shows its filename and error chip without an include control"),
    ("failed_excluded", "failed documents start excluded from document text"),
    ("analysis_only_success", "the actual analysis request receives only successful document text"),
    ("escaped_error", "document error messages cannot insert HTML"),
    ("all_failed", "an all-error batch stays visible without opening the clinical reveal gate"),
]:
    check(description, browser_result.get(key, False), browser_error)


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
