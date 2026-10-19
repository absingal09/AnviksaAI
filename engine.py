"""engine.py — triage. No model call, no network, no randomness, no arithmetic
anyone could mistake for a measurement.

    from engine import Engine
    eng = Engine.load()
    out = eng.assess(candidates, findings, confirmed_exclusions=["Pulmonary embolism"])

propose.py says what conditions are worth keeping in mind. This decides how
they are shown: which band of harm-if-missed they fall into, how well the
findings actually support them, which are still open, and what single
question would most divide the list. It runs offline in milliseconds.

WHAT WAS DELETED FROM THIS FILE, AND WHY IT MUST NOT COME BACK

`posterior()`, `tau`, `abstain_entropy`, `_entropy`, `ranked_risk()` and the
information-gain block are gone, along with numpy. They implemented naive
Bayes over 49 synthetic pathologies from a dataset that has been dropped.

Naive Bayes scored each condition by multiplying findings independently, so
nothing anywhere could represent findings that only mean something TOGETHER.
Pericarditis demonstrated it concretely: "worse lying flat, better leaning
forward" plus "recent viral illness" is a classic trio, but scored separately
the viral-illness term dragged toward the dozen commoner respiratory
illnesses in the set and pericarditis never surfaced. No rephrasing of the
input fixed it, because the architecture had nowhere to put the joint
meaning. That is an argument about structure, not about tuning, which is why
the replacement is a model that reads findings together and a file that rates
harm — not a better prior.

THE THREE RULES THIS FILE ENFORCES

1. NO NUMBER EVER REACHES THE SCREEN. Not a percentage, not a score, not a
   confidence. Severity numbers exist in severity.yaml because a file needs
   sortable values; they are converted to words here and the payload carries
   only words. Nobody is ever certain in medicine, and three significant
   figures imply a precision no input to this system can support.

2. SEVERITY AND SUPPORT ARE TWO AXES AND ARE NEVER MULTIPLIED. `P x severity`
   was meaningful when P was a real probability. Without one, a single
   blended number is a fabrication that throws away the two things the
   clinician actually wants held apart: how bad would this be to miss, and
   how well does it actually fit this patient.

3. EXCLUSION PROPOSES; IT NEVER AUTO-REMOVES. Recording a CTPA makes the
   panel ASK whether to mark pulmonary embolism ruled out. It does not
   silently drop it. Every other error in this system adds noise to a list a
   human is already reading. An exclusion error REMOVES A WARNING, and nobody
   ever sees what they were not shown. It is the only failure mode here
   shaped like patient harm, so a human makes every removal.
"""

import re
from pathlib import Path

import yaml

import vocabulary

# severity.yaml stores orders of magnitude, not a linear 1-5 — missing an MI
# is about a hundred times worse than missing a sore throat, not five times.
# These are the words those magnitudes are shown as. The numbers stay in this
# module; the payload carries the words.
BANDS = [
    (100, "critical", "Lethal or irreversible within hours if missed"),
    (30, "serious", "Life- or organ-threatening over days"),
    (10, "significant", "Significant harm; needs definite treatment"),
    (3, "routine", "Needs treatment, not urgent"),
    (1, "self-limiting", "Usually settles on its own"),
]

UNRATED = "unrated"
UNRATED_NOTE = "Not in the reviewed severity file — shown, but unrated"

# A condition reaches Can't-Miss at this severity or above. severity.yaml
# guarantees every condition at or above it has an exclusion rule, so
# everything that can reach the panel can also leave it. A panel that cannot
# be finished becomes a smoke alarm that goes off when you make toast.
CANT_MISS_FLOOR = 10

# Words that qualify a condition without identifying it. Stripped before
# matching so "Acute pulmonary embolism" still finds "Pulmonary embolism".
# Kept short on purpose: every word removed here is a word that can no longer
# tell two conditions apart, so this list stays at hedges and tempo markers
# and never grows to include anything anatomical or pathological.
_NOISE = {
    "possible", "probable", "suspected", "likely", "query", "rule", "out",
    "acute", "subacute", "chronic", "early", "late", "initial", "recurrent",
    "and", "or", "with", "of", "the", "a", "an", "in", "to", "due", "secondary",
}


# Words that describe a condition without naming one. Excluded from the
# distinctive-word rule in match() — never from containment, where they carry
# real meaning ("Viral pharyngitis" is not "pharyngitis").
#
# Each of these happened to appear in exactly one rulebook entry by accident,
# which made it look identifying when it is not. Observed: "Viral exanthem"
# matched "Viral pharyngitis" on the word `viral` alone, rating a rash as a
# sore throat, and "Stevens-Johnson syndrome / drug reaction" matched "Acute
# dystonic reaction" on `reaction`. Both are now honestly unrated.
_GENERIC = {
    # kind-of-illness words
    "viral", "bacterial", "fungal", "infection", "infective", "infectious",
    "reaction", "disease", "disorder", "syndrome", "failure", "attack",
    "episode", "illness", "condition", "injury", "pain", "fever",
    "neoplasm", "tumour", "tumor", "cancer", "bleed", "bleeding", "crisis",
    "exacerbation", "obstruction", "detachment", "retention",
    # where it is, not what it is. "Lower respiratory tract infection"
    # matched "Urinary tract infection" on `tract`, and "Upper respiratory
    # tract infection" matched "Upper GI bleed" on `upper` — the second at
    # the top severity band.
    "upper", "lower", "left", "right", "anterior", "posterior", "proximal",
    "distal", "bilateral", "unilateral", "tract", "chest", "abdominal",
    "cardiac", "renal", "hepatic", "gastric", "systemic", "deep", "vein",
    "media", "angle", "closure", "gi", "pre", "post",
    # organ adjectives. Each names a site that many conditions share, so on
    # its own it identifies nothing: "Temporal lobe epilepsy" matched
    # "Temporal arteritis", "Urinary retention" matched "Urinary tract
    # infection", and left unblocked "Diabetic neuropathy" would have been
    # rated as diabetic ketoacidosis and "Thyroid nodule" as thyroid storm —
    # both at the top severity band. Every entry these appear in is still
    # reachable through containment, which needs the whole name.
    "temporal", "urinary", "ovarian", "testicular", "inguinal", "pancreatic",
    "retinal", "atrial", "thyroid", "diabetic", "kidney", "bowel", "adrenal",
    "ectopic", "pregnancy", "intracranial", "cerebral", "coronary",
    # degree and manner
    "severe", "mild", "moderate", "stable", "unstable", "spontaneous",
    "localised", "localized", "raised", "pressure", "signs", "warning",
    "food", "human", "virus", "tachycardia", "bradycardia",
}


def _normalise(text):
    """Lowercase, strip anything parenthesised, fold British spellings.

    The spelling fold matters more than it looks: severity.yaml is written in
    British medical English (haemorrhage, oedema, anaemia, ischaemia) and a
    model will freely return either. A missed match on that alone would push
    subarachnoid haemorrhage off Can't-Miss.
    """
    text = re.sub(r"\([^)]*\)", " ", text.lower())
    text = text.replace("ae", "e").replace("oe", "e")
    return re.sub(r"[^a-z0-9/ ]+", " ", text)


def _tokens(text):
    return {w for w in _normalise(text).replace("/", " ").split() if w and w not in _NOISE}


def _alternatives(name):
    """A severity.yaml name split on '/' into the alternatives it offers."""
    return [
        {w for w in part.split() if w and w not in _NOISE}
        for part in _normalise(name).split("/")
    ]


class Engine:
    def __init__(self, severity, excluded_by, aliases=None):
        self.severity = severity
        self.excluded_by = excluded_by
        # Another name for the SAME rulebook entry, so "Systemic lupus
        # erythematosus flare" finds the entry written as "SLE". Aliases use
        # containment matching only, never the loose distinctive-word rule.
        self.aliases = dict(aliases or {})
        # token -> the single rated condition whose name contains it, for
        # tokens appearing in exactly one. See match(), rule 2.
        #
        # BUILT FROM CANONICAL NAMES ONLY — never from aliases. Expanding an
        # acronym injects generic words into this index that the rulebook
        # never actually used to name anything: expanding PSVT put
        # "tachycardia" in here, and "Postural orthostatic tachycardia
        # syndrome" promptly matched PSVT. Expanding URTI would have made
        # "Lower respiratory tract infection" match the upper one.
        #
        # So the two rules draw on different vocabularies, deliberately.
        # Aliases are exact alternative names and are matched STRICTLY, by
        # containment. The loose rule only ever fires on words the rulebook
        # itself chose, which are the words a clinician actually reviewed.
        owners = {}
        for name, value in severity.items():
            if value is None:
                continue
            for token in _tokens(name):
                if token in _GENERIC:
                    continue
                owners.setdefault(token, set()).add(name)
        self._unique_token = {t: next(iter(n)) for t, n in owners.items() if len(n) == 1}

    @classmethod
    def load(cls, severity_path=None):
        path = Path(severity_path or Path(__file__).parent / "severity.yaml")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        # `default: null`, deliberately. An unlisted condition is UNRATED, not
        # benign. The previous file defaulted to 3, which meant anything the
        # rulebook had never heard of was quietly treated as mildly important
        # and sorted below things it might well outrank.
        return cls(
            severity=dict(raw.get("pathologies") or {}),
            excluded_by=dict(raw.get("excluded_by") or {}),
            aliases=dict(raw.get("aliases") or {}),
        )

    # ------------------------------------------------------------- severity
    def match(self, condition):
        """Find this condition's entry in severity.yaml. -> (name, rank) or None.

        THIS IS A SAFETY MECHANISM, NOT A CONVENIENCE. The candidate list is
        unbounded by design — propose.py is given no list to choose from — so
        conditions arrive named however the model chose to name them. Matching
        on exact strings meant "Panic attack (acute anxiety episode)" scored as
        unrated when severity.yaml plainly rates "Panic attack", which was
        observed on the canonical case the first time this ran end to end.

        Harmless there. Not harmless in the other direction: "Acute pulmonary
        embolism" failing to match "Pulmonary embolism" would drop a critical
        condition out of Can't-Miss entirely, losing its exclusion rule with
        it, over a spelling. That is a warning removed by a string comparison,
        which is precisely what this system is not allowed to do.

        So matching is deliberately generous, and generous in the safe
        direction. Over-matching puts something on the panel a clinician can
        dismiss in a second. Under-matching hides it. Where several entries
        match, the most severe wins, for the same reason.

        What it does NOT do is guess. Nothing here maps an unfamiliar condition
        onto a familiar-looking one — matching requires the rulebook entry's
        own words to be present in the candidate's name. A genuinely unlisted
        condition still comes back unrated and is shown as such, because the
        rulebook not knowing something is a fact worth putting on screen.
        """
        rank = self.severity.get(condition)
        if rank is not None:
            return condition, rank

        wanted = _tokens(condition)
        if not wanted:
            return None

        best = None

        def consider(name, value):
            nonlocal best
            if best is None or value > best[1]:
                best = (name, value)

        for name, value in self.severity.items():
            if value is None:
                continue
            # Rule 1 — containment. The rulebook entry's words all appear in
            # the candidate's name: "Acute pulmonary embolism" contains
            # "Pulmonary embolism". "Possible NSTEMI / STEMI" and "Acute
            # stroke / TIA" name alternatives rather than one long title, so
            # either side matching counts.
            matched = False
            for source in (name, self.aliases.get(name, "")):
                if not source:
                    continue
                for alt in _alternatives(source):
                    if alt and alt <= wanted:
                        consider(name, value)
                        matched = True
                        break
                if matched:
                    break

        # Rule 2 — a distinctive shared word. Containment alone fails whenever
        # the RULEBOOK is the more specific of the two: "Meningococcal
        # meningitis" does not contain "Bacterial meningitis", and scored
        # unrated — dropping a condition at the top severity band off
        # Can't-Miss and losing its exclusion rule with it. Same for "Dengue
        # infection" against "Dengue with warning signs". Both found by
        # running a real consultation, not by reading the code.
        #
        # A word that appears in exactly ONE rulebook entry identifies that
        # entry: "meningitis", "dengue", "embolism", "dissection". A word
        # appearing in several — "pulmonary" spans embolism, oedema and
        # neoplasm — identifies nothing and is deliberately powerless here.
        #
        # This over-matches sometimes: "heat stroke" shares "stroke" with
        # "Acute stroke / TIA" and will be rated as one. That is the safe
        # direction and it is visible — the row shows which rulebook entry was
        # applied — whereas the failure it replaces was a critical condition
        # quietly demoted to unrated.
        for token in wanted:
            owner = self._unique_token.get(token)
            if owner is not None:
                consider(owner, self.severity[owner])

        return best

    def _rank(self, condition):
        """Raw severity, or None if unrated. Internal only — never returned in
        a payload; the payload carries the band word."""
        found = self.match(condition)
        return found[1] if found else None

    def band(self, condition):
        """-> (band_name, plain_english_note). Words, never numbers."""
        rank = self._rank(condition)
        if rank is None:
            return UNRATED, UNRATED_NOTE
        for floor, name, note in BANDS:
            if rank >= floor:
                return name, note
        return BANDS[-1][1], BANDS[-1][2]

    # -------------------------------------------------------------- support
    @staticmethod
    def support(candidate):
        """How well the findings actually fit — the OTHER axis. Words only,
        and never combined with severity.

        Counting supporting findings is a blunt instrument and is meant to
        be. A weighted score would look more sophisticated and would be
        inventing weights nobody has measured. The clinician reads the
        reasoning; this only decides ordering within a band.
        """
        for_it = len(candidate.get("supported_by") or [])
        against = len(candidate.get("opposed_by") or [])
        if against > for_it:
            return "mostly against"
        # Net, not "any opposition at all". An earlier version demanded zero
        # opposing findings for "strong", which read eight supporting findings
        # and one denial as merely moderate — a description no clinician
        # looking at the same list would recognise.
        net = for_it - against
        if net >= 3:
            return "strong"
        if net >= 2:
            return "moderate"
        if net >= 1:
            return "limited"
        return "no findings for it"

    _SUPPORT_ORDER = ["strong", "moderate", "limited", "no findings for it", "mostly against"]

    # ------------------------------------------------------------ exclusion
    def exclusion_options(self, condition):
        """What would take this condition off the panel — as {token, label}.

        Returned so the panel can ASK. Nothing here removes anything; the
        caller passes confirmed exclusions back in on the next call and they
        persist for the rest of the consult.
        """
        # Looked up under the MATCHED rulebook name, not the name the model
        # returned. Otherwise "Acute pulmonary embolism" reaches Can't-Miss
        # via match() and then offers no way off it — a condition that can be
        # shown but never cleared, which is exactly the un-finishable panel
        # the exclusion rules exist to prevent.
        found = self.match(condition)
        name = found[0] if found else condition
        return [
            {"action": token, "label": vocabulary.INVESTIGATIONS.get(token, token)}
            for token in self.excluded_by.get(name, [])
        ]

    # ------------------------------------------------------------- question
    def next_question(self, candidates, findings, asked=()):
        """The one finding worth asking about next.

        NOT "the best question" — that label was on the previous build and it
        overclaimed. Information gain is optimal only with respect to a proxy
        objective over a distribution, and there is no distribution here.
        Nobody can have the best question. This picks a useful one and says so.

        The rule, in one sentence: ask about the finding that bears on the most
        dangerous still-open candidate; where several do, take the one that
        candidate itself ranked as most decisive; break what remains toward the
        finding fewer candidates are waiting on, since a question everything
        hinges on cannot tell those things apart.

        Ordering by severity first is deliberate and is not the forbidden
        blend. The prohibition is on multiplying severity by support into one
        displayed figure that pretends to be a measurement. Choosing which
        question to spend a clinician's ten seconds on is a different act, and
        "the one that could change what's on Can't-Miss" is the only ordering
        consistent with why this product exists.

        The rank within a candidate's `unresolved` list is the model's own
        judgment of which answer would most change whether that condition
        belongs — information already produced, so this uses it rather than
        inventing a second opinion about it.

        NEVER RE-ASKS. A finding already present, already absent, or already
        put to the patient is out. This is what a stable vocabulary buys: the
        previous build could not match a whole-question denial against its own
        asked set, so the identical question came back every single turn.
        """
        answered = set(findings) | set(asked)
        if not candidates:
            return None

        # fid -> (candidates waiting on it, best rank any of them gave it)
        waiting = {}
        for cand in candidates:
            for position, fid in enumerate(cand.get("unresolved") or []):
                if fid in answered or fid not in vocabulary.FINDINGS:
                    continue
                cands, best_rank = waiting.get(fid, ([], position))
                cands.append(cand)
                waiting[fid] = (cands, min(best_rank, position))
        if not waiting:
            return None

        def score(item):
            fid, (cands, best_rank) = item
            worst = max((self._rank(c["condition"]) or 0) for c in cands)
            # Negated where smaller is better, so one max() reads correctly.
            # The final term is alphabetical and exists only so the same input
            # always produces the same question — there is no principle in it,
            # and pretending otherwise would be the kind of false rigour this
            # file exists to keep out.
            return (worst, -best_rank, -len(cands), [-ord(ch) for ch in fid])

        fid, (cands, _) = max(waiting.items(), key=score)
        return {
            "finding": fid,
            "label": vocabulary.label(fid),
            "ask": vocabulary.question(fid),
            # Examination and observation items are for the clinician. They
            # are legitimate next steps but must never be spoken aloud to the
            # patient, and the caller cannot tell from the text alone.
            "clinician_only": vocabulary.is_clinician_only(fid),
            "divides": sorted(c["condition"] for c in cands),
        }

    # -------------------------------------------------------------- assess
    def assess(self, candidates, findings, confirmed_exclusions=(), asked=(), other_findings=()):
        """One call -> the whole payload. Contains no numbers, by construction.

        candidates:            propose.propose() output
        findings:              {finding_id: True|False}; anything absent is UNKNOWN
        confirmed_exclusions:  conditions a CLINICIAN has confirmed ruled out.
                               Persists across every update for the rest of the
                               consult — this and answered findings are the only
                               two things that survive; everything else here
                               recomputes from scratch each time.
        asked:                 findings already put to the patient but not yet
                               answered, so they are not asked again.
        other_findings:        separate positive evidence used only for support
                               labels, never catalog state, questions or exclusions.
        """
        def exclusion_key(condition):
            found = self.match(condition)
            return ("rated", found[0]) if found else ("unrated", condition.strip().casefold())

        confirmed = {exclusion_key(condition) for condition in confirmed_exclusions}
        other_labels = {
            item["id"]: item["text"] for item in other_findings
            if isinstance(item, dict) and isinstance(item.get("id"), str)
            and re.fullmatch(r"other:\d+", item["id"])
            and isinstance(item.get("text"), str) and item["text"].strip()
        }

        def supplemental_id(value):
            return isinstance(value, str) and value.lower().startswith("other:")

        # Keep ephemeral citations out of negative/unknown evidence even if
        # a caller bypasses proposal validation. Do not alter catalog state.
        candidates = [dict(cand,
            supported_by=[f for f in cand.get("supported_by") or []
                          if not supplemental_id(f) or f in other_labels],
            opposed_by=[f for f in cand.get("opposed_by") or [] if not supplemental_id(f)],
            unresolved=[f for f in cand.get("unresolved") or [] if not supplemental_id(f)],
        ) for cand in candidates]
        proposed = {exclusion_key(c["condition"]) for c in candidates}
        for condition in sorted(set(confirmed_exclusions)):
            key = exclusion_key(condition)
            if key not in proposed:
                # Confirmation is consultation state even when the model stops
                # proposing this condition. Do not fabricate clinical reasoning
                # or retain patient state on the shared Engine instance.
                candidates.append({
                    "condition": condition,
                    "reasoning": "Confirmed ruled out by the clinician during this consultation.",
                    "supported_by": [], "opposed_by": [], "unresolved": [],
                })
                proposed.add(key)

        # Abstain visibly. No candidates means say so — never fall back to
        # something generic styled as though it came from this patient.
        if not candidates:
            return {
                "abstaining": True,
                "message": (
                    "Nothing was extracted from this consultation yet."
                    if not findings and not other_labels else
                    # Rare, and it should be. propose.py casts wide on thin
                    # findings rather than staying silent — a vague complaint
                    # is the moment a clinician most needs the broad list,
                    # because it is before anything has been ruled out. This
                    # message means the findings carried no clinical content
                    # at all, not that they were merely nonspecific.
                    "Nothing here yet that names a condition. Keep recording."
                ),
                "cant_miss": [], "differential": [], "unrated": [],
                "ruled_out": [], "doesnt_fit": [], "next_question": None,
                "findings_present": [], "findings_absent": [],
            }

        cant_miss, differential, unrated, ruled_out = [], [], [], []

        for cand in candidates:
            condition = cand["condition"]
            found = self.match(condition)
            rank = found[1] if found else None
            band, note = self.band(condition)
            entry = {
                "condition": condition,
                "reasoning": cand["reasoning"],
                # Which severity.yaml entry was applied, when it is not the
                # name on the row. The clinician can then see that "Panic
                # attack (acute anxiety episode)" was rated as "Panic attack"
                # — the one place a generous match could go wrong is the one
                # place it must be visible.
                "rated_as": found[0] if (found and found[0] != condition) else None,
                # The two axes, side by side, never merged into one figure.
                "severity_band": band,
                "severity_note": note,
                "support": self.support(cand),
                # isinstance guard, not decoration: propose.py filters these
                # to real ids, but a malformed candidate reaching here must
                # render oddly rather than raise. A crash in triage shows the
                # clinician an empty panel, and an empty panel is
                # indistinguishable from "nothing to worry about".
                "supported_by": [
                    {"finding": f, "label": other_labels[f] if f in other_labels else vocabulary.label(f)}
                    for f in cand.get("supported_by") or [] if isinstance(f, str)
                ],
                "opposed_by": [
                    {"finding": f, "label": vocabulary.label(f)}
                    for f in cand.get("opposed_by") or [] if isinstance(f, str)
                ],
            }

            if exclusion_key(condition) in confirmed:
                # Kept and shown, not deleted. The clinician needs to see that
                # it was considered and cleared — that is what lets the panel
                # be finished rather than merely emptied.
                ruled_out.append(entry)
            elif rank is None:
                # Unrated is a visible tier, never a silent drop. The rulebook
                # not knowing a condition is a fact about the rulebook.
                unrated.append(entry)
            elif rank >= CANT_MISS_FLOOR:
                entry["exclusion_options"] = self.exclusion_options(condition)
                cant_miss.append(entry)
            else:
                differential.append(entry)

        # Order within a band by how well the findings support it. Across
        # bands, severity leads — but the two values stay separate in the
        # payload and are never combined into one figure.
        def order(rows):
            rows.sort(key=lambda r: (
                -(self._rank(r["condition"]) or 0),
                self._SUPPORT_ORDER.index(r["support"]),
                r["condition"],
            ))
            return rows

        # Findings the patient gave that NOTHING on the list explains. The
        # "Doesn't Fit" panel: the patient told you this and no candidate
        # accounts for it. Cheap to compute and it points at the gap rather
        # than at the list.
        explained = {
            f["finding"]
            for row in cant_miss + differential + unrated + ruled_out
            for f in row["supported_by"]
        }
        # `onset` findings are excluded: they qualify other findings rather
        # than standing alone. "Started within hours" being unexplained is not
        # a gap in the list, it is a timing note, and letting it through fills
        # this panel with noise that hides the one entry that matters.
        doesnt_fit = [
            {"finding": f, "label": vocabulary.label(f)}
            for f, state in sorted(findings.items())
            if state is True
            and f not in explained
            and vocabulary.FINDINGS.get(f, {}).get("system") != "onset"
        ]

        live = [c for c in candidates if exclusion_key(c["condition"]) not in confirmed]

        return {
            "abstaining": False,
            "message": None,
            "cant_miss": order(cant_miss),
            "differential": order(differential),
            "unrated": order(unrated),
            "ruled_out": order(ruled_out),
            "doesnt_fit": doesnt_fit,
            "next_question": self.next_question(live, findings, asked),
            "findings_present": [
                {"finding": f, "label": vocabulary.label(f)}
                for f, s in sorted(findings.items()) if s is True
            ],
            "findings_absent": [
                {"finding": f, "label": vocabulary.label(f)}
                for f, s in sorted(findings.items()) if s is False
            ],
        }


if __name__ == "__main__":
    import json
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    eng = Engine.load()
    print(json.dumps(eng.assess([], {}), indent=2))
