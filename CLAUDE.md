# AnviksaAI

A recall aid for a 2-minute Indian OPD consultation. It does **not** diagnose,
and it does **not** tell you how likely anything is. It reads what the patient
actually said and brought, proposes what is worth not forgetting, flags which
of those are catastrophic if missed, and asks the one question most likely to
tell the conditions apart.

The problem it solves is not "the doctor doesn't know medicine." It is: in the
ninety seconds available, with a queue outside, a clinician cannot hold
everything the patient offered in working memory at once. Something gets
dropped. Occasionally the thing that gets dropped is the thing that kills them.

Medicine reaches certainty through tests and confirmation. We are upstream of
that: we only make sure nothing dangerous silently fell off the list first.

## Why this isn't a chat wrapper

The question every reviewer will ask: how is this different from a doctor
pasting the case into ChatGPT? The honest answer has two halves, and the
second one matters.

**Nobody types a prompt.** To ask an AI "here's what my patient has", you
must already know what to include — and the failure being addressed is
*forgetting*. If you remembered the calf swelling well enough to type it, you
did not need the tool. Input here is ambient: a recording of a conversation
that was happening anyway, and a photo of a document the patient brought.
Chat answers the question you thought to ask; this surfaces what you didn't.

**Three things a chat structurally cannot do.** Track exclusion state across a
consult, so the panel can be *finished* rather than re-listing PE on every
chest complaint forever. Avoid re-asking what was already answered. And
refuse to name the diagnosis — ask a model what the patient has and it tells
you; the refusal is the product.

**Severity is not the model's opinion.** The axis deciding what gets flagged
as lethal is a reviewed file, not a generation that drifts between runs.

**And the uncomfortable half.** The candidate list comes from the same kind of
model — if ChatGPT would name aortic dissection, so will this. The reasoning
shown is model-generated. With no eval, **this cannot claim to be more
accurate than a doctor using ChatGPT**, and nobody should say otherwise.

The honest claim is narrow: it happens without being asked, it keeps state a
chat cannot, and the dangerousness ranking is human-authored. Every one of
those is a rule below. Erode any of them and the honest answer to this
question becomes "we aren't different" — which is why they are written as
non-negotiable rather than preferences.

## Architecture

```
Sarvam captures  ->  LLM reads and proposes  ->  our rulebook triages  ->  clinician decides
```

| Stage | Owner | What |
|---|---|---|
| capture | `capture.py` | `saaras-v3` batch STT (diarized), `sarvam-vision` on documents |
| screen | `extract.py` | is this document clinical at all? per document, before anything is merged |
| extract | `extract.py` | this patient's transcript/documents -> findings, three-valued |
| propose | `propose.py` | findings -> candidate conditions + what supports/opposes each |
| triage | `engine.py` | severity band, exclusion state, support ordering |
| ask | `engine.py` | the finding that bears on the most dangerous open candidate |
| speak | `capture.speak()` | the question, translated, in the patient's language |

There is no `api/` package; every module above sits at the repo root.

**Screening happens per document and cannot be moved later.** `/analyze`
concatenates every document into one blob, and after that a game
screenshot's words are indistinguishable from a prescription's. That is not
a hypothetical either: a screenshot uploaded alongside a real consultation
changed the candidate list and added a finding nothing the patient said
supported. So the screen runs in `/capture`, the last point at which
per-document text exists.

`screen_document()` judges what a document IS. It never sees the transcript
or the vocabulary — a screen that knew which findings were being looked for
would start rating documents by whether they were useful rather than by
whether they were real. And it flags rather than deletes: see the exclusion
rule below, which it is a case of.

Two LLM calls, never one. Extraction must not see the candidate list, or the
model starts extracting findings *because* they fit a condition it is already
considering. That is not hypothetical — an earlier provider fabricated
`E_110` (immobility) on a transcript that never mentioned it, and this
project switched providers over exactly that failure.

## Non-negotiable rules

**No number ever reaches the screen.** Not a percentage, not a score, not a
confidence. Nobody is ever 100% sure in medicine, and three significant
figures imply a precision no input here can support. Show a severity band and
a support strength, in words.

**No language model ever assigns severity or exclusion.** The LLM proposes
candidates and says what supports them. `severity.yaml` — hand-written,
clinician-reviewed — decides what is dangerous and what rules it out. Ask a
model "how bad is it to miss this" and the answer drifts between runs, and
there is no artifact anyone can audit.

**Exclusion proposes; it never auto-removes.** When the clinician records a
CTPA, the panel *asks* "mark PE as ruled out?" — it does not silently drop it.
Every other error in this system adds noise to a list a human is already
reading. An exclusion error *removes a warning*, and nobody ever sees what
they were not shown. That is the only failure mode here shaped like patient
harm, so a human makes every removal.

**Severity and support are two axes. Never multiply them.** `P x severity` was
meaningful when `P` was a real probability. Without one, a single blended
number is a fabrication that discards information the clinician wants
separately: how bad if missed, and how well does this actually fit.

**Findings need stable names.** Extraction maps to a fixed vocabulary we
author. Free-form findings cannot be matched across turns, so "don't re-ask
what was already answered" silently fails — which produced a real
stuck-on-the-same-question bug (a categorical code's whole-question denial
never landed in the asked set, so the identical question returned forever).

**Every candidate carries its reasoning.** "Aortic dissection — because
tearing pain radiating to the back, sudden onset." With no dataset and no
eval, the visible reasoning and the clinician *are* the entire safety net.
A suggestion whose justification cannot be read cannot be checked.

**Three-valued findings: present / absent / UNKNOWN.** Never guess absent for
something nobody asked about. An unasked symptom is not a denied symptom, and
conflating them penalises exactly the conditions that would have caused it.

**`temperature=0` and a fixed seed, always.** Confirmed live: identical input
returned different findings run to run until these were pinned, which changed
what was on screen for reasons that had nothing to do with the patient. Vary
the seed only on retry, never on the first attempt.

**Cache every model call on generation.** Sarvam vision is capped at 10-30
req/min on every tier. Handle `429` with exponential backoff. Never call in a
render loop.

**Evidence arrives in pieces, and re-extraction is over the whole thing.**
A consult is not one input — it is a recording, then a document, then another
recording, each landing minutes apart. Extraction re-runs across the FULL
accumulated transcript and document text every time, never incrementally over
just the new chunk. A later statement can correct an earlier one ("no, it's
the right calf, not the left"), and only a model seeing both can resolve
that; merging per-chunk findings would keep both as true. This costs a full
call per update. Pay it.

**Two things survive across updates; everything else recomputes.**
*Confirmed exclusions* — once the clinician ticks "PE ruled out by CTPA",
that holds for the rest of the consult and must be sent with every subsequent
call. *Answered findings* — which is only possible because the vocabulary
gives findings stable ids; this is precisely what failed before, when a
whole-question denial could not be matched and the same question returned
forever. Candidates, ordering, and the next question are recomputed fresh
each time.

**The panel must not churn.** A list that reshuffles on every sentence is
unreadable, and worse, the clinician loses track of what they already
considered. Show what changed rather than silently reordering.

## What was deleted, and why it must not come back

**DDXPlus, entirely.** 49 synthetic pathologies, 223 evidence codes. Deleted
because a tool whose job is "what catastrophic thing have you not excluded"
was running on a dataset containing no aortic dissection, no subarachnoid
haemorrhage, no meningitis, no sepsis, no ectopic pregnancy, no torsion, no
cauda equina — while including Ebola and Chagas. Its base rates were foreign
to an Indian OPD, and its statistics were generated by a simulator, not
observed in patients.

**Naive Bayes, the posterior, `tau`, `abstain_entropy`.** Naive Bayes scores
each condition by multiplying findings independently — nothing anywhere could
represent findings that only mean something *together*. Pericarditis
demonstrated this concretely: "pain worse lying flat, better leaning forward"
plus "recent viral infection" is a classic pair, but scored separately the
viral-infection term dragged toward the dozen more common respiratory
illnesses in the set, and pericarditis never surfaced. No rephrasing fixed
it, because the architecture had nowhere to put the joint meaning.

**Information gain and "best question."** Optimal only with respect to a
proxy objective over a synthetic distribution — the label overclaimed. The
replacement is honest: the finding whose support most *divides* the live
candidate set. Nobody can have the best question.

**Calibration and accuracy claims.** With no ground truth there is no eval,
no `tune.py`, no measured accuracy. This is the deliberate price of unbounded
coverage. Do not claim a number that cannot be measured.

## UI rules

- **No chat interface.** Ever. Nobody prompts anything during a 2-minute
  consult, and it is the tell that this is a wrapper.
- **The reveal gate.** Differential and Can't-Miss stay hidden until BOTH: the
  clinician has committed a note, AND at least one real live input has
  happened (a recording sent, or a document scanned — either satisfies it).
- **Abstain visibly.** When nothing was extracted, say so. Never show a
  generic prior-driven suggestion styled as though it came from this patient.
- **Severity-unrated is a visible tier.** A proposed condition absent from
  `severity.yaml` still appears, marked unrated. It is never silently dropped.
- **A rejected document is set aside, never discarded.** Its text is kept,
  the screen's reading of it is shown ("mobile game screenshot"), and one
  click puts it back. Same asymmetry as exclusion: throwing away a real
  prescription loses evidence nobody will ever know was missing.
- **Cast wide when the findings are thin.** A vague complaint is the moment
  *before* anything has been ruled out, which is when the broad list is worth
  most. Support strength exists so a weak fit can be shown as weak — refusing
  to speak throws that mechanism away and is not the cautious option, just an
  unhelpful one. Abstention is for input with no clinical content at all.
- **Never speak a question without showing what was said.** The spoken
  sentence is printed beside the button. See "Later, not now".

## Layout

```
capture.py         Sarvam STT / vision / translation, disk-cached by content hash
llm.py             the one place that talks to the chat model: client, retry, cache
vocabulary.py      loads findings.yaml, once
extract.py         transcript/documents -> findings (fixed vocabulary, three-valued)
propose.py         findings -> candidates + supporting/opposing findings + reasoning
engine.py          severity band, exclusion state, support ordering, next question
severity.yaml      harm-if-missed + exclusion rules   <- clinical judgment, hand-written
findings.yaml      the finding vocabulary             <- hand-written
cases/            regression cases; _template.yaml describes the schema and is skipped
selftest.py        offline invariants and expressibility checks for every case
run_cases.py       manual live extract/propose/triage checks; needs SARVAM_API_KEY
.github/workflows/selftest.yml  Python 3.12 offline suite on pushes and pull requests
main.py            FastAPI: /capture (STT/OCR + screen), /analyze (extract+propose+triage), /speak
web/index.html     single file, no build step
render.yaml        the free deployment; the key is entered in the
                   dashboard, never here
```

`llm.py` is deliberately ignorant of medicine — it takes a system prompt, a
user message and a tool schema. A shared helper that started assembling
clinical prompts would be the first step back toward one merged call.

**Keep `/capture` and `/analyze` separate.** They look mergeable and are not.
`/capture` does STT or OCR only; `/analyze` does extract + propose + triage
over everything accumulated so far. When the frontend used one combined
endpoint, every recording and every upload ran a full extraction whose result
was thrown away, then ran a second one — double the cost and double the
latency on every single action. Merging them back reintroduces that.

## The canonical case

`cases/` replaces the old DDXPlus-coded demo fixtures. The original case lives
unchanged at `cases/pe_masquerading_as_panic.yaml`; `cases/_template.yaml`
describes the schema without adding another clinical case. Both runners skip
underscore-prefixed files. Nothing renders from these fixtures or preloads them
into the UI.

`python selftest.py` stays offline and checks each case's vocabulary references
alongside the existing rulebook invariants. With the single PE case it still
runs all 94 checks. GitHub Actions runs this suite on every push and pull request
with Python 3.12 and no API key.

`python run_cases.py` is a separate, manual live runner requiring
`SARVAM_API_KEY`. `python run_cases.py pe_masquerading_as_panic` selects one case
by ID. It feeds transcript and document text through extraction, proposal, and
engine triage, then prints PASS/FAIL for findings, candidates, the next question,
and numeric-output restrictions (plus reasoning and exclusion expectations).
It reuses the existing caches, so unchanged cases with the same model/prompts
make no new calls while their responses are cached. Neither the offline suite
nor CI invokes it. These regressions are not measured clinical accuracy.

The case that has always exercised this product properly: 34F, sudden chest
tightness and dyspnoea, a photographed strip revealing a combined oral
contraceptive, and a long journey mentioned in passing. The doctor is
thinking panic attack. **Pulmonary embolism must appear on Can't-Miss**, and
the next question should be about unilateral calf swelling. Anything that
fails to surface PE here is broken, whatever else it gets right.

Verified live end to end on 2026-08-23: extraction picked up both ambient
items — the bus journey buried mid-answer in a reply about something else,
and the contraceptive read off the blister strip, which the patient never
mentions and is never asked about. Neither is something anyone would have
thought to type into a chat window, which is the argument for this product
stated in one case. PE came back critical with strong support, panic attack
stayed visible in the differential rather than being argued with, and the
next question was the calf.

The `must_be_unknown` list in that file matters as much as the expectations.
Nobody asked this patient about her calves, so `calf_swelling_unilateral`
must come back UNKNOWN, not absent. An extractor that guesses absent for
unasked findings penalises pulmonary embolism specifically, on the exact case
pulmonary embolism has to survive.

## Migration status (2026-08-23) — done

The DDXPlus migration is complete. Every step below has landed and its
acceptance test passes. `python selftest.py` runs 94 invariants offline in
about a second, and a fresh clone runs with no external download — which had
never been true before, because `extract.py` used to read its catalog from a
gitignored 172 MB dataset.

1. ~~Rewrite this file~~
2. ~~Rewrite `severity.yaml`~~ — 91 conditions, 75 panel-eligible, every one
   with an exclusion rule (0 missing, 0 orphans).
3. ~~Author `findings.yaml`~~ — 217 findings, all binary, plus labels for all
   83 exclusion actions. The original case, now in
   `cases/pe_masquerading_as_panic.yaml`, was authored alongside it; offline
   invariants and the manual `run_cases.py` now read the `cases/` folder.
4. ~~Write `propose.py`, rewrite `extract.py`~~ — two calls, `llm.py` shared
   between them, both cached at `temperature=0`.
5. ~~Strip `engine.py`~~ — no probability is computed anywhere; numpy gone.
6. ~~Rewrite `selftest.py`~~
7. ~~Rewrite the panels~~ — verified in the browser: not one digit renders in
   any panel.
8. ~~Delete the DDXPlus remnants~~ — requirements down from 11 packages to 8.

**The one thing that did not land, and never will by writing code:**
`severity.yaml` still has not been reviewed by a clinician. It is drafted from
published can't-miss lists, and with no eval downstream it is the only
safeguard in the system. That review gates real use, not the code.

### What the migration itself taught, that was not in the plan

**Matching candidate names to `severity.yaml` by exact string is a safety
hole.** Found by running the canonical case end to end for the first time:
the model returned "Panic attack (acute anxiety episode)" and it scored
*unrated*. Harmless there. But "Acute pulmonary embolism" failing to match
"Pulmonary embolism" would have dropped a critical condition off Can't-Miss
and taken its exclusion rule with it — a warning removed by a string
comparison, which is the one thing this system must never do.

The candidate list is unbounded by design, so names arrive however the model
chose to write them. `engine.match()` is now token-based, folds British
spellings (`haemorrhage`/`hemorrhage` alone would have lost subarachnoid
haemorrhage), handles the `/` alternations, and is deliberately generous in
the safe direction: over-matching puts something on the panel a clinician
dismisses in a second, under-matching hides it. Where several entries match,
the most severe wins. The applied rulebook entry is shown on the row whenever
it differs from the name proposed, because the one place a generous match
could go wrong is the one place it has to be visible.

Do not "tighten" this to exact matching. It will look cleaner and it will
silently demote critical conditions.

The same defect kept reappearing in new shapes once real consultations were
run through it, and all of them are pinned in `selftest.py` now:

- the rulebook being the MORE specific of the two — "Meningococcal
  meningitis" does not contain "Bacterial meningitis", so it scored unrated;
- a word unique to one entry *by accident* identifying nothing — "Viral
  exanthem" matched "Viral pharyngitis" on `viral`, a rash rated as a sore
  throat;
- acronym against expansion — "Systemic lupus erythematosus flare" shares no
  word at all with an entry named `SLE`, which is why `severity.yaml` grew an
  `aliases` block;
- and expanding those acronyms leaking generic words back into the loose
  rule, so "Upper respiratory tract infection" matched **"Upper GI bleed"** on
  `upper`, at the top severity band.

The resolution worth keeping: **aliases feed strict containment only; the
loose distinctive-word rule draws solely on the rulebook's own canonical
names** — the words a clinician will actually review. Words for where
something is or how bad it is never identify a condition alone.

**Two more things silently ate information, and neither looked like a bug.**

`propose.py` matched finding ids exactly, so a model answering "Headache"
instead of "headache" lost every supporting finding. The panel then showed
real candidates each captioned "no findings for it" while the same findings
sat in Doesn't Fit saying nothing accounted for them — both halves of the
screen wrong at once. Resolve loosely, filter strictly, and print what gets
dropped rather than swallowing it.

And a cache key must cover **everything that determines the answer**, not
just the patient's words. Keyed on `(model, transcript, doc_text)`, adding
three findings to the vocabulary changed nothing for any transcript already
seen; the old, smaller extraction was served forever. Caught only by
re-running the exact transcript that had exposed the gap. The key now hashes
the system prompt, the tool schema and the whole catalog — and the catalog,
not just its ids, because the prompt carries each finding's question too.

## Later, not now

Recorded so they aren't re-derived. Nothing here is a todo, and nothing here
is blocking — except where it says it is.

**Read exclusions off the report instead of asking for a tick.** The
clinician photographs the d-dimer result; the document already flows through
`/capture`. Proposing "mark PE ruled out?" from the report itself closes the
loop that currently needs manual input, and it is the single highest-value
thing after the migration.

**The note-versus-transcript check.** The committed note and the full
transcript both already exist in the same session, and nothing compares them.
*"You wrote ?anxiety — the patient also mentioned a long journey and is on a
combined contraceptive."* That is the anchoring bias this product exists to
interrupt, stated directly, and it needs no new inputs.

**Local prevalence as an explicit prior.** An Indian OPD is not a Swiss ED.
Dengue, TB, enteric fever and rheumatic heart disease sit at wholly different
base rates. This is the one place local data genuinely helps — as a stated,
reviewable prior, not resurrected population statistics.

**Speak the question in the patient's language — properly this time.** This
was always billed as a must and was never built. The old build shipped
`HINDI_QUESTIONS`, a dict containing exactly one hardcoded Hindi sentence for
one question id; every other language, and Hindi for any other question, fell
through to speaking the raw English in an English voice. It went unnoticed
because the scripted walkthrough pinned the displayed question to that one id,
so the fallback never fired in a demo.

**This is now built** — `POST /speak`, translate then `capture.speak()`,
with two guards. The route refuses a question it could not translate, because
falling back to English in an English voice is the old bug wearing a new
coat. And it refuses clinician-only findings server-side, not in the
frontend, so no caller can read "Are the neck veins distended?" to a patient.

**What remains is the verification, and it is not optional.** The translated
sentence is printed on screen beside the audio, and that mitigation earned
itself on the first test: asked in Bengali, the calf question came back as
*"is one of your child's milk ducts more swollen than the other?"* — Sarvam
had read "calf" in the animal sense. Nothing else in this system would have
caught it. The wording was changed to "lower leg" and is now correct in four
languages, but that is one question out of 217, fixed after the fact.

A mistranslated question asked aloud produces an answer that enters the
findings pipeline as genuine evidence, indistinguishable afterwards from
something the patient said. Printing the translation makes that catchable by
someone who reads the language; it does not make it safe. **A native speaker
still has to go through all 217 `ask` strings** — same standard as
`severity.yaml`, for the same reason.

**Measurement.** The deepest gap. Everything above is unfalsifiable until a
clinician scores real consults on whether the panel was useful or noise. This
is what separates a working demo from a tool anyone should rely on, and no
amount of engineering substitutes for it.

## Don't build

Auth, accounts, a real database, medical imaging, any model trained from
scratch, a mobile app, more than one demo language, or any accuracy claim
that isn't measured — which, now, is all of them.

Specifically tempting and specifically wrong: generating `severity.yaml` with
an LLM (it deletes the only audit surface), and putting percentages back
because they look authoritative.
