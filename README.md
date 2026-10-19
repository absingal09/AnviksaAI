# AnviksaAI

**[Try it →](https://anviksaai.onrender.com/web/)**  ·  decision support only, not a
medical device, not for clinical use. `severity.yaml` has not been reviewed by
a clinician — see [What this is not](#what-this-is-not) before reading anything
on the panel as advice.

A recall aid for a 2-minute Indian OPD consultation. It does **not** diagnose,
and it does **not** tell you how likely anything is.

It listens to the consultation that was happening anyway, reads the documents
the patient brought, proposes what is worth not forgetting, flags which of
those are catastrophic if missed, and asks the one question most likely to
tell them apart.

The problem it solves is not "the doctor doesn't know medicine." It is that in
the ninety seconds available, with a queue outside, nobody can hold everything
the patient offered in working memory at once. Something gets dropped.
Occasionally the thing that gets dropped is the thing that kills them.

## Run it

```bash
pip install -r requirements.txt
```

Put a Sarvam API key in `.env` as `SARVAM_API_KEY=...`, then:

```bash
python main.py
```

Open <http://127.0.0.1:8000/web/>. No dataset download, no build step, no
database. The invariant suite runs offline in about a second:

```bash
python selftest.py
```

GitHub Actions runs this offline suite on every push and pull request with
Python 3.12 and no API key. It checks every `cases/*.yaml` fixture except files
whose names start with an underscore. The original PE case is now in
`cases/pe_masquerading_as_panic.yaml`; `cases/_template.yaml` documents the schema
for future reviewed cases and is skipped by both runners.

Live case checks are separate and manual. With `SARVAM_API_KEY` configured:

```bash
python run_cases.py
python run_cases.py pe_masquerading_as_panic
```

`run_cases.py` sends each case's transcript and document text through the same
extract → propose → triage sequence as `/analyze`, reporting PASS/FAIL for the
expected findings and panel behavior. It uses the existing model-call cache;
rerunning unchanged cases with the same model/prompts costs no new calls while
the cached responses remain available. Missing credentials produce a clear
message. Neither `selftest.py` nor CI runs this live runner. Case regressions do
not establish clinical accuracy, and fixtures never populate the app's UI.

## Deploy it, free

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/absingal09/AnviksaAI)

`render.yaml` is a Render blueprint. Push the repo to GitHub, then in the
Render dashboard: **New → Blueprint**, pick the repo, and it reads the file.
It will ask for one value — `SARVAM_API_KEY`. **You type that, in Render's
dashboard, and it never enters the repo.** A key committed to a public repo
is scraped within hours, and deleting the commit does not undo it; rotating
in the Sarvam dashboard is the only real fix.

The button reads the same file, and needs the repo to be **public** — on a
private repo it only works for someone who already has Render's GitHub App
pointed at it, which for a stranger means it silently does nothing. Either
way it stops at the same place: it will ask *you* for `SARVAM_API_KEY`, and
whoever clicks it deploys under their own account with their own key.

The choice of host is not aesthetic. **This app holds an HTTP request open
for minutes** — `/analyze` makes two chat calls (~244s worst case) and
`/capture` waits on a batch STT job (up to 180s). Serverless function
platforms cap a response at 10-60s and will return an error while the work
is still running, which the panel shows as *nothing happening*. Render web
services allow 100 minutes.

What the free plan costs, so it isn't discovered later:

| | |
|---|---|
| Sleeps after 15 idle minutes | the next request pays a 30-60s cold start |
| Ephemeral filesystem | `cache/` is empty after every sleep and redeploy |
| 750 instance-hours / month | across the whole workspace |

An external pinger on `/web/` every 5 minutes removes the sleep. That costs
~720 of the 750 hours, so it fits alone and does not fit alongside a second
free service. Render tolerates this rather than supporting it.

Hugging Face Spaces looks like the obvious alternative and is not one twice
over. `CPU Basic — FREE` on its pricing page is the *hourly hardware* cost; a
Docker Space still requires PRO to create, the free-account exception being
two Gradio Spaces on ZeroGPU. And its proxy returns 504 at around 60s, so
even paid it would cut `/analyze` off mid-call.

Nothing breaks when the cache is wiped — every call falls through to a real
one. The first consult after a wake just pays full price for work it had
already paid for once.

**The deployed app has no authentication.** Anyone with the URL can spend
your Sarvam quota. That is acceptable for showing it to people and is not
acceptable for anything else.

## How it works

```
Sarvam captures  ->  an LLM reads and proposes  ->  our rulebook triages  ->  clinician decides
```

| File | What |
|---|---|
| `capture.py` | `saaras-v3` batch STT (diarized), `sarvam-vision` on documents, translation, `bulbul-v3` speech. All disk-cached by content hash. |
| `extract.py` | transcript + documents → findings, three-valued, each with the quote that justifies it. Also screens each document: is this clinical at all? |
| `propose.py` | findings → candidate conditions + what supports and opposes each. Never sees the transcript. |
| `engine.py` | severity band, exclusion state, support ordering, the next question. No network, no randomness. |
| `findings.yaml` | the finding vocabulary — hand-written |
| `severity.yaml` | harm-if-missed + what rules each condition out — **hand-written clinical judgment** |
| `cases/` | regression fixtures, including the original PE case and an underscore-prefixed schema template |
| `selftest.py` | offline invariants and fixture expressibility checks; no API key |
| `run_cases.py` | manual live case runner, with optional case ID and existing API caches |
| `.github/workflows/selftest.yml` | runs the offline suite on every push and pull request with Python 3.12 |
| `main.py` | FastAPI. `POST /capture` (STT/OCR + screening), `POST /analyze` (extract + propose + triage), `POST /speak` (translate + say the question aloud) |
| `web/index.html` | single file, no build step |

Two LLM calls, never one. Extraction must not see the candidate list, or the
model starts extracting findings *because* they fit a condition it is already
considering — which is not hypothetical, it is why this project changed
providers once already.

## The rules it is built on

These are not preferences. Each one is written up at length in `CLAUDE.md`,
and each exists because breaking it produced a real bug or a real piece of
false precision at some point in this project's history.

- **No number ever reaches the screen.** No percentage, no score, no
  confidence. A severity band and a support strength, in words.
- **No language model assigns severity or exclusion.** `severity.yaml` does,
  and a human wrote it.
- **Exclusion proposes; it never auto-removes.** Every other error here adds
  noise to a list a human is reading. An exclusion error removes a warning,
  and nobody ever sees what they were not shown.
- **Severity and support are two axes and are never multiplied.**
- **Three-valued findings: present / absent / UNKNOWN.** An unasked symptom is
  not a denied symptom.
- **Every candidate carries its reasoning.** With no eval, the visible
  reasoning and the clinician are the entire safety net.
- **No chat interface.** Nobody types a prompt during a 2-minute consult.
- **A document that doesn't look clinical is set aside, not deleted.** One
  click puts it back — discarding a real prescription loses evidence nobody
  would know was missing.
- **A spoken question is always shown as text.** Machine translation of
  clinical phrasing is unverified; printing it is what makes a bad one
  catchable.

## What this is not

**There is no eval, no calibration, and no measured accuracy.** There is no
ground truth to measure against, which is the deliberate price of an unbounded
candidate list. `selftest.py` passing means the rules still hold — it says
nothing about whether the panel is useful.

The candidates come from a general-purpose language model. If a chat assistant
would name aortic dissection on a given presentation, so will this. **This
cannot claim to be more accurate than a doctor typing the case into a chat
window**, and nobody should say otherwise. What it can claim: it happens
without anyone being asked to type a prompt, it keeps state across a consult
that a chat cannot, and the dangerousness ranking applied to its output is a
reviewed file rather than a generation.

**The 217 questions in `findings.yaml` have not been checked by native
speakers.** They are translated at the moment they are spoken. The first test
of that feature asked, in Bengali, whether one of the patient's *child's milk
ducts* was swollen — "calf" read as the baby animal. That one is fixed; the
other 216 are unverified, and a mistranslated question produces an answer
that enters the pipeline as genuine evidence.

**`severity.yaml` has not been reviewed by a clinician.** It is drafted from
published emergency-medicine can't-miss lists and is the only human-auditable
safeguard in the system. That review is the blocker no amount of code clears.

Decision support only. Not a medical device. Not for clinical use.

## Reading further

`CLAUDE.md` is the real document: the architecture, every non-negotiable rule
and why it exists, what was deleted and why it must not come back, and what is
deliberately deferred.
