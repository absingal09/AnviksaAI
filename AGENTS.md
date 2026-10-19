# AnviksaAI: instructions for Codex

Use this guide when working anywhere in this repository. It adapts the project
knowledge and constraints in `CLAUDE.md` for Codex, with setup and command details
checked against `README.md` and the code. Preserve the reasons behind the rules;
they record failures that have already happened.

## Project purpose and limits

AnviksaAI is a recall aid for a two-minute Indian outpatient department (OPD)
consultation. It reads an ambient consultation recording and photographs of the
documents a patient brings, proposes conditions worth remembering, flags those
that would be catastrophic to miss, and suggests a question that distinguishes
the open candidates. The clinician decides. The product does not diagnose or
estimate how likely a condition is.

The problem is limited working memory under time pressure, not a clinician's
lack of medical knowledge. Evidence can be spread across an incidental remark,
a medicine strip, and a complaint. The tool operates before tests and
confirmation, to keep dangerous possibilities from silently falling off the list.

Keep these distinctions from a generic chat assistant intact:

- Input is ambient. Nobody has to remember what to include in a typed prompt;
  remembering to type a forgotten clue would defeat the purpose.
- Consultation state tracks confirmed exclusions and answered findings. A panel
  must be finishable, and must not repeatedly ask an answered question or keep
  re-listing an excluded condition.
- The tool refuses to name a diagnosis as its conclusion.
- Dangerousness comes from a human-authored, auditable severity file rather than
  a model's changing opinion.

The candidate list and its reasoning still come from a general-purpose model.
There is no evaluation, calibration, ground-truth dataset, or measured accuracy.
Do not claim superiority to a doctor using ChatGPT, or treat passing invariants
as evidence that the panel is useful. The honest claims are ambient capture,
consultation state, and a human-authored dangerousness rulebook.

**Decision support only; not a medical device and not for clinical use.** Despite
references in code to a "reviewed" rulebook, `severity.yaml` is a draft from
published can't-miss lists and has not been reviewed by a clinician. That review
gates real use. Native-speaker review of the question translations and clinician
assessment of real consultations are also outstanding; engineering cannot
substitute for them.

## Install and run

Run commands from the repository root. `.python-version` specifies Python 3.12.
An isolated virtual environment is recommended:

```sh
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell, then install dependencies:

```sh
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `SARVAM_API_KEY`. The same credential is
used for STT, vision, chat, speech synthesis, and translation. The optional
task settings default to `gemma4` for screening and `glm5.3` for independent
extraction/proposal calls, using Sarvam V2 (beta access required). Configure
`SARVAM_SCREEN_MODEL`, `SARVAM_EXTRACT_MODEL`, and `SARVAM_PROPOSE_MODEL`
independently. The legacy `SARVAM_CHAT_MODEL` overrides all defaults; task settings
take precedence. `sarvam-105b` remains a V1 option for keys without V2 access.
Never silently change models on a provider failure. Never commit real credentials. If a key
is committed, rotate it in Sarvam's dashboard; deleting it does not remove it
from Git history.

```sh
python main.py
```

Open `http://127.0.0.1:8000/web/`. There is no dataset download, frontend build
step, or database. The server defaults to `0.0.0.0` and port `8000`, reads `PORT`
from the environment, and accepts `--host` and `--port`, for example:

```sh
python main.py --host 127.0.0.1 --port 8080
```

The running app and nonempty capture/analysis/speech requests need network access
and a valid Sarvam key. `cache/` stores generated responses locally and is
gitignored. A lost cache causes fresh paid calls rather than incorrect results.

## Tests and verification

```sh
python selftest.py
```

This is the repository's offline invariant suite. After dependencies are
installed it needs no API key, external dataset, or model calls, and normally
runs in about a second. It uses handwritten candidate fixtures, not live model
output. It checks vocabulary and exclusion integrity, severity floors, name
matching, readable reasoning, unrated candidates, absence of numeric scores,
three-valued/empty-input behavior, no repeated questions, human-confirmed
exclusion, and deterministic triage. Run it for changes to these behaviors.

`cases/` contains clinical regression fixtures. `selftest.py` loops over every
`*.yaml` file there except underscore-prefixed files, checking vocabulary and
rulebook properties offline. The original PE fixture is unchanged at
`cases/pe_masquerading_as_panic.yaml`; with this one case all 94 existing checks
still run. `.github/workflows/selftest.yml` runs the suite on every push and pull
request with Python 3.12, installing `requirements.txt` and needing no API key.

Live checks are separate and manual, requiring `SARVAM_API_KEY`:

```sh
python run_cases.py
python run_cases.py pe_masquerading_as_panic
```

`run_cases.py` uses the same extract → propose → triage sequence as `/analyze`,
checking present/absent/unknown findings, candidate placement, the next question,
numeric output, reasoning, and exclusion options. Existing caches avoid new API
calls for unchanged cases with the same model/prompts while cached responses
remain available. Missing credentials produce a clear message. Never invoke it
from `selftest.py` or CI; do not run live cases without authorization and a key.
The underscore-prefixed `cases/_template.yaml` is schema guidance, not another
clinical case, and both runners skip it. These checks do not establish clinical
accuracy. For UI changes, also check the reveal gate, state across updates,
document restoration, and displayed spoken text in the browser. No linter,
formatter, or frontend build configuration is checked in.

## Repository structure

All application Python modules live at the root; there is no `api/` package.

| Path | Responsibility |
| --- | --- |
| `capture.py` | Sarvam `saaras:v4` diarized batch STT, `sarvam-vision` OCR, pinned `sarvam-translate:v1`, and Hindi/English `bulbul:v4-flash` speech (v3 for other existing languages); model-aware content-hash caches and retries. |
| `llm.py` | Shared chat client, forced structured tool calls, timeout/retry handling, deterministic sampling, and disk cache helpers. |
| `vocabulary.py` | Loads `findings.yaml` once; exposes IDs, labels, questions, investigation labels, and clinician-only checks. |
| `extract.py` | Independently screens each document, then maps transcript/documents to fixed-vocabulary findings and supporting quotes. |
| `propose.py` | Findings to unbounded candidate conditions, reasoning, supporting/opposing findings, and unresolved findings; never sees the transcript. |
| `engine.py` | Offline deterministic severity bands, rulebook matching, exclusion state, support ordering, and next-question selection. |
| `severity.yaml` | Handwritten harm-if-missed values, exclusion rules, and matching aliases; clinical judgment, not generated model output. |
| `findings.yaml` | Handwritten stable finding vocabulary, labels, question strings, and labels for exclusion actions. |
| `cases/` | Human-readable regression fixtures, including the original PE case and a skipped `_template.yaml`; never a UI data source. |
| `selftest.py` | Offline invariants and expressibility checks for every case; no API key. |
| `run_cases.py` | Manual live case runner; optional case ID, existing caches, and explicit API key required. |
| `.github/workflows/selftest.yml` | Python 3.12 offline suite on pushes and pull requests. |
| `main.py` | FastAPI routes, request models, pipeline orchestration, and static frontend serving. |
| `web/index.html` | Entire frontend: HTML, CSS, and JavaScript in one file, without a build step. |
| `requirements.txt` | Python runtime dependencies. |
| `.python-version`, `.env.example` | Python version and documented environment variables. |
| `render.yaml`, `Procfile` | Deployment configuration; both start `python main.py`. |
| `README.md`, `CLAUDE.md` | Public overview/setup and the original detailed project guidance. |
| `.claude/launch.json` | Existing development launch configuration for port 8000. |
| `cache/` | Generated, gitignored API response/audio caches. |

## Architecture and boundaries

```text
Sarvam captures -> LLM reads and proposes -> rulebook triages -> clinician decides
```

1. `POST /capture` accepts an audio file for STT, or documents for OCR and
   per-document screening. It does not extract clinical findings or propose
   candidates.
2. `POST /analyze` receives the full accumulated transcript and included document
   text, plus `confirmed_exclusions` and `asked`. It runs extraction, proposal,
   and deterministic engine assessment.
3. `POST /speak` accepts a finding ID and language, looks up the question,
   translates it, calls `capture.speak()`, and returns both audio and the text
   actually spoken.
4. `/web/` serves only the frontend directory. Never mount the repository root;
   doing so would expose `.env` and `.git/`.

Keep `/capture` and `/analyze` separate. A former combined route extracted once
for every recording/upload, discarded that result, then extracted again on
panel refresh, doubling cost and latency. Do not restore that architecture.
Run blocking provider work in the threadpool from async routes; blocking the
event loop previously froze the server and refused new connections.

Screen documents individually in `/capture`, before their text is concatenated
for analysis. A game screenshot once changed the candidate list and introduced
an unsupported finding when mixed into a real consultation. Once texts are
merged, their origins cannot be separated. `screen_document()` judges what a
document *is*, without seeing the transcript or vocabulary; otherwise it starts
judging usefulness instead of authenticity. It flags documents rather than
deleting them, and a screening failure keeps the document with a visible
"not screened" status.

Extraction and proposal are two separate clinical model calls, with separate
prompts. Extraction must not see candidate conditions, and proposal only sees
findings. An earlier provider fabricated `E_110` (immobility) to fit a condition
despite no mention in the transcript; that failure caused a provider change.
Screening is its own additional model call. Keep `llm.py` ignorant of medicine:
it takes a system prompt, user content, and tool schema, and returns parsed
arguments. Do not move clinical prompt assembly into that shared helper.

Chat models are selected per task in `llm.config()`. Gemma screening uses a small
nonreasoning budget; GLM/DeepSeek reasoning needs a larger shared thinking/tool
budget. The installed SDK predates V2 chat, so `llm.py` uses HTTPX for V2 and the
SDK for V1. Preserve forced tool calls, changed retry seeds, explicit reasoning
settings and model/profile-aware cache keys. Never turn malformed tool output
into a successful empty extraction. The offline suite tests both transports.

Capture defaults and overrides are documented in `.env.example`:
`SARVAM_STT_MODEL`, `SARVAM_TRANSLATION_MODEL`, `SARVAM_TTS_MODEL`. TTS `auto`
uses documented, live-verified v4 Hindi/English personas; other existing
languages keep v3 pending voice verification. Malayalam/Odia have no native v4
personas currently. Do not substitute an English persona for another language.
Hash the selected capture model and speech settings into caches as well.

The 2026-10-19 single-case model comparison is recorded in README's Models
section. GLM and DeepSeek each passed 27/28 live PE expectations, leaving
`prior_panic_attacks` unknown. This is an unresolved live miss, not a passing
clinical evaluation. Do not alter the fixture or infer absence to hide it.

## Non-negotiable behavior

- **No number ever reaches the screen.** No percentages, probabilities,
  scores, or confidence values. Show severity bands and support strength as
  words. Internal sortable severity values must not become displayed precision
  or numeric panel fields.
- **The model never assigns severity or exclusion.** `severity.yaml` owns both.
  Keep it handwritten, auditable, and subject to clinician review. A model's
  answer about dangerousness drifts between runs and is not an audit artifact.
- **Exclusion proposes; a human confirms every removal.** Recording a CTPA may
  prompt "mark PE as ruled out?"; it must not silently drop the warning. An
  exclusion error removes information the clinician cannot know was hidden.
  Preserve the ruled-out row and reasoning so the decision remains auditable.
- **Keep severity and support as separate axes.** Never multiply them or add a
  blended risk score. Without real probabilities, `P x severity` fabricates a
  measurement and discards useful distinctions.
- **Use stable authored finding IDs.** Free-form findings cannot be matched
  across turns. A former categorical whole-question denial failed to enter the
  asked set and caused the same question to recur indefinitely.
- **Findings are three-valued.** Present is `True`, explicitly absent is `False`,
  and unmentioned/unknown findings are omitted from the finding map. Never infer
  absence because nobody asked. Never invent evidence to fit a candidate.
- **Every candidate carries readable reasoning.** Preserve the evidence and
  extraction quotes that make it checkable. With no evaluation dataset, visible
  reasoning and clinician judgment are the safety net.
- **Use `temperature=0` and a fixed initial seed.** Identical inputs previously
  produced different findings and panels. Change the seed only on retries;
  `llm.py` uses the attempt number, starting at zero. An identical retry seed
  previously reproduced missing tool calls indefinitely.
- **Cache every model call when generated.** Use content-based keys, retry
  `429` responses with exponential backoff, and never trigger calls in a render
  loop. Existing guidance records a vision limit of 10-30 requests per minute;
  avoid repeated scans of identical content. Preserve transient chat retries
  for missing tool calls, provider errors, and unwrapped HTTPX network/timeouts.
- **Re-extract the full accumulated evidence on every update.** A consultation
  arrives in pieces; a later correction can supersede an earlier statement.
  Extracting only the new chunk and merging finding maps could keep both sides
  of a correction as true. Pay for the full call and cache repeats.
- **Carry state across updates deliberately.** Confirmed exclusions and
  answered/already-asked finding IDs survive and are sent on every analysis.
  Candidates, ordering, and the next question recompute from current evidence.
  Do not repeatedly ask something already present, absent, or asked.
- **Keep the panel stable.** Show what changed instead of silently reshuffling
  rows on every sentence; clinicians must retain their place in the list.

## UI requirements

- No chat interface or prompt-entry workflow.
- Keep Differential and Can't-Miss hidden until both a clinician note has been
  committed and at least one real recording has been sent or document scanned.
  The other reasoning panels follow the same gate in the current frontend.
- Abstain visibly when no clinical findings were extracted. Never fill empty
  input with generic prior-driven candidates dressed as patient-specific output.
  Denials alone do not supply positive evidence for a candidate.
- Cast wide for thin but genuine clinical findings. A vague complaint precedes
  exclusions; broad candidates with visibly weak support are useful. Do not
  confuse weak evidence with no clinical content.
- Show conditions absent from `severity.yaml` in an explicit **unrated** tier;
  never silently drop them.
- Set rejected documents aside with their full text retained and the screen's
  description visible, such as "mobile game screenshot". One click restores
  them. Wrongly discarding a prescription hides evidence just like a bad
  exclusion hides a warning.
- Show the exact spoken/translated question beside its audio control. Refuse
  failed translation rather than silently speaking English. Enforce the
  clinician-only question restriction server-side, not just in the browser.
- Never show prerecorded case output, an offline replay, or a precomputed
  contract payload as if it came from the current patient.

## Coding and data conventions

Follow the existing small-module style: four-space Python indentation,
`snake_case` functions and variables, uppercase constants, and explanatory
docstrings/comments for architectural constraints and regression causes. Keep
route/request handling in `main.py`, provider plumbing in `capture.py`/`llm.py`,
and deterministic clinical presentation rules in `engine.py`.

Use UTF-8 for text/YAML/JSON and preserve Indian-language text. Load shared
vocabulary through `vocabulary.py`, whose import-time validation catches duplicate
IDs. Finding IDs must be unique lowercase `snake_case` across all systems, with
a readable label and `ask` question. Exclusion actions need readable labels in
the investigations mapping; they are clinician actions, not patient symptoms.
Questions beginning `(Examination)` or `(Observation)` are clinician-only by
convention; preserve those prefixes and the server-side check.

Keep every panel-eligible severity entry excludable, leave the default severity
null so unknown conditions are honestly unrated, and validate aliases against
real canonical rulebook entries. Preserve deliberate British/American medical
spelling normalization. Do not add dependencies or a frontend toolchain without
a concrete need; the current app has no build step.

## Regression lessons to preserve

**Condition names cannot require exact matches.** The canonical live case once
returned "Panic attack (acute anxiety episode)", which became unrated. The same
bug on "Acute pulmonary embolism" would hide a critical condition from Can't-Miss
and lose its exclusion rule. `engine.match()` normalizes tokens and British
spellings, handles slash alternatives, and favors the most severe entry when
multiple entries match. Show the actual rulebook entry when it differs from
the proposed name. Do not replace this with exact string matching.

Generous matching also needs its existing guards. Keep these regression shapes:

- A more specific rulebook entry can match a broader proposed name; the
  "Meningococcal meningitis" / "Bacterial meningitis" case exposed this.
- A word unique by accident does not identify a disease: "Viral exanthem" must
  not match "Viral pharyngitis" solely on `viral`.
- Acronyms need explicit aliases: "Systemic lupus erythematosus flare" cannot
  find `SLE` by shared words alone.
- Alias expansions must not leak generic words into loose matching: "Upper
  respiratory tract infection" previously matched "Upper GI bleed" on `upper`,
  incorrectly receiving the highest severity band.

Aliases feed **strict containment only**. The loose distinctive-word rule uses
only canonical rulebook names. Location, severity, and other generic qualifiers
must never identify a condition alone. These cases are pinned in `selftest.py`.

**Resolve evidence IDs tolerantly, then filter strictly.** A proposal containing
"Headache" instead of `headache` previously lost all support. Candidates then
said "no findings for it" while Doesn't Fit claimed those same findings were
unexplained. Accept real IDs case-insensitively and known labels; do not guess
unknown findings. Supporting findings must actually be present, opposing ones
absent, and unresolved ones genuinely unknown. Log dropped references rather
than swallowing them.

**Cache keys must cover everything that determines the answer.** Model plus
patient text alone served obsolete extractions after vocabulary changes. Hash
the system prompt, tool schema, and relevant complete catalog, including labels
and question wording rather than just IDs, along with the model and input.
Otherwise changes can appear ineffective for every consultation already cached.

## Canonical case

`cases/` replaces the old DDXPlus-based demo fixtures. The original canonical
case is `cases/pe_masquerading_as_panic.yaml`; `_template.yaml` documents the
schema and is skipped. `selftest.py` reads these fixtures offline, while
`run_cases.py` is a manual live check. Humans also review the expected panel
description. Nothing renders from these files and no walkthrough preloads them.

The case is a 34-year-old woman with sudden chest tightness and dyspnoea. A
photographed blister strip reveals a combined oral contraceptive; a long bus
journey is mentioned in passing. The clinician is thinking panic attack.
Pulmonary embolism **must appear on Can't-Miss**, with reasoning, and panic
attack must remain visible in the differential. The next question should concern
unilateral calf swelling or calf pain/tenderness. Failure to surface PE here is
a regression regardless of other successes.

The `must_be_unknown` list matters as much as the positive expectations. Nobody
asked about her calves, so `calf_swelling_unilateral` must remain UNKNOWN, not
absent; guessing absent specifically penalizes the condition this case must
retain. Honor all the file's expected findings and unknowns.

Historical live verification on 2026-08-23 extracted both ambient clues: the
journey buried in another answer and the contraceptive read from packaging that
the patient never named. PE was critical with strong support, panic attack
remained visible, and the next question concerned the calf. This is the product
argument in one case, not a measured accuracy claim.

## Deleted approaches and completed migration

Do not restore these removed designs:

- **DDXPlus:** its 49 synthetic pathologies and 223 evidence codes omitted
  aortic dissection, subarachnoid haemorrhage, meningitis, sepsis, ectopic
  pregnancy, torsion, and cauda equina while including Ebola and Chagas. Its
  simulated statistics and foreign base rates were unsuitable for an Indian
  OPD recall tool. No external dataset should be needed for a fresh clone.
- **Naive Bayes, posterior probabilities, `tau`, `abstain_entropy`, and numpy:**
  independent finding multiplication could not represent joint meaning.
  Pericarditis was missed when positional pain plus recent viral illness was
  split into terms favoring common respiratory illnesses. Rephrasing could not
  fix that structural limitation.
- **Information gain and a claim to the "best question":** optimizing a proxy
  objective over synthetic data did not justify that label. The replacement
  asks about a finding whose support divides the live candidates, prioritizing
  dangerous open conditions; do not claim universal optimality.
- **Calibration/accuracy claims or `tune.py`:** without ground truth there is no
  evaluation. Unbounded candidate coverage deliberately trades away the old
  synthetic measurement story.

The DDXPlus migration was recorded complete on 2026-08-23: guidance and severity
rules were rewritten, the vocabulary and canonical case authored, two cached
model calls separated through `llm.py`, the probability engine removed, tests
rewritten, panels made free of numeric scores, and dataset remnants deleted.
The historical snapshot reported 91 conditions, 75 panel-eligible conditions
with no missing/orphan exclusions, 217 binary findings, 83 labeled exclusion
actions, 94 offline invariants, and dependencies reduced from 11 to 8. These
are migration records, not counts to hardcode into new behavior or current
verification claims. A fresh clone no longer depended on the gitignored
172 MB catalog. Clinician review of `severity.yaml` remains outstanding.

## Deployment constraints

`render.yaml` is the existing Render blueprint: install requirements, start
`python main.py`, serve from Singapore, and health-check `/web/`. Set the Sarvam
key in the hosting dashboard; `sync: false` keeps it out of the blueprint.
`Procfile` uses the same entry point. Keep Python version management in
`.python-version`.

Preserve long-request support. The code budgets approximately 244 seconds for
the pair of chat calls and up to 180 seconds for batch STT; the browser analysis
timeout is 360 seconds to include a sleeping host's cold start. Coordinate
provider retries, frontend abort timing, and proxy limits. The repository's
deployment notes chose a long-lived Render web service because short serverless
timeouts cut work off while it is still running. Free-host caches are ephemeral,
so waking/redeploying can repeat API costs. The deployed app has no
authentication; anyone with its URL can consume the configured Sarvam quota.

## Deferred ideas and review gates

These are recorded design directions, not an instruction to implement them or a
new task list. Review gates are explicitly identified:

- **Propose exclusions from reports:** photographs already pass through
  `/capture`; use their contents to propose a clinician confirmation instead of
  requiring manual initiation. This was identified as the highest-value next
  feature. It must still never auto-remove warnings.
- **Compare the committed note with the transcript:** both exist in-session.
  Highlight omitted evidence, for example a note saying `?anxiety` despite a
  journey and contraceptive, to expose anchoring without needing new inputs.
- **Explicit local prevalence:** Indian OPD prevalence for dengue, TB, enteric
  fever, and rheumatic heart disease could inform a stated, reviewable prior.
  Do not revive unexamined synthetic population statistics.
- **Speech is implemented; verification remains a gate.** `/speak` translates
  then synthesizes, refuses failed translations, and blocks clinician-only
  findings server-side. Do not restore the old `HINDI_QUESTIONS` shortcut: it
  had one hardcoded Hindi question and silently spoke English for everything
  else, hidden by a scripted demo pinned to that question.
- **Native-speaker review is required.** A Bengali translation read "calf" as
  the baby animal and asked about a child's milk ducts. Changing that question
  to "lower leg" was checked in four languages; it does not verify the rest of
  the 217 question strings recorded at migration. A bad spoken question creates
  an answer that later looks like genuine clinical evidence. Displaying the
  translation makes errors catchable, not safe; native speakers still need to
  review all question strings.
- **Clinical measurement is required.** Clinicians must score real consultations
  for useful recall versus noise. No engineering result replaces that work or
  the still-missing clinical review of `severity.yaml`.

## Out of scope

Do not build authentication, accounts, a real database, medical imaging, a model
trained from scratch, a mobile app, more than one demo language, or any
unmeasured accuracy claim. Existing speech support does
not authorize expanding the demo's language scope. In particular, never
generate `severity.yaml` with an LLM or put percentages back on the panel to
make it look authoritative.
