"""Manual live case checks; never imported by selftest.py or run by CI.

    python run_cases.py
    python run_cases.py pe_masquerading_as_panic

Requires SARVAM_API_KEY. Extraction and proposal use their existing disk
caches, so unchanged inputs, model and prompts reuse prior model responses.
Inputs are already transcribed/OCR text, just as for main.py's /analyze.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent
PANELS = ("cant_miss", "differential", "unrated", "ruled_out")


def load_cases():
    """Load actual case files, rejecting malformed fixtures before API calls."""
    cases = []
    seen = set()
    for path in sorted((ROOT / "cases").glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        try:
            case = yaml.safe_load(path.read_text(encoding="utf-8"))
            case_id = case["case"]["id"]
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError("case.id must be a nonempty string")
            if case_id in seen:
                raise ValueError(f"duplicate case.id: {case_id}")
            for key in ("transcript", "document_text"):
                if not isinstance(case["input"][key], str):
                    raise ValueError(f"input.{key} must be a string")
            for group, keys in (
                ("expected_findings", ("present", "absent", "must_be_unknown")),
                ("expectations", ("must_appear_on_cant_miss", "must_appear_somewhere", "next_question_in")),
            ):
                for key in keys:
                    values = case[group][key]
                    if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                        raise ValueError(f"{group}.{key} must be a list of nonempty strings")
            for key in ("every_candidate_has_reasoning", "no_number_in_output", "cant_miss_entries_are_excludable"):
                if key in case["expectations"] and not isinstance(case["expectations"][key], bool):
                    raise ValueError(f"expectations.{key} must be true or false")
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        seen.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError("No case fixtures found in cases/ (underscore-prefixed files are skipped)")
    return cases


def number_issues(value, path="output"):
    """Find numbers or score fields in the public engine payload.

    Private evidence quotes may legitimately contain doses or durations and
    are not displayed by the clinician panels, so they are excluded here.
    """
    issues = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.startswith("_"):
                continue
            if key in {"p", "probability", "score", "confidence", "entropy", "risk", "weighted", "priority_score"}:
                issues.append(f"{path}.{key}")
            issues.extend(number_issues(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(number_issues(item, f"{path}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        issues.append(path)
    elif isinstance(value, str) and re.search(r"\d", value):
        issues.append(path)
    return issues


def has_condition(rows, expected, engine):
    """Use the app's rulebook matching for generated naming variants."""
    expected_match = engine.match(expected)
    for row in rows:
        if row["condition"].casefold() == expected.casefold():
            return True
        actual_match = engine.match(row["condition"])
        if expected_match and actual_match and expected_match[0] == actual_match[0]:
            return True
    return False


def check_expectations(case, findings, payload, engine):
    """Print each expectation separately and return whether all passed."""
    results = []

    def check(label, passed, detail=""):
        results.append(bool(passed))
        print(f"  {'PASS' if passed else 'FAIL'} {label}" + (f" — {detail}" if detail and not passed else ""))

    expected = case["expected_findings"]
    for group, state in (("present", True), ("absent", False)):
        for fid in expected[group]:
            actual = "present" if findings.get(fid) is True else "absent" if findings.get(fid) is False else "unknown"
            check(f"{group}: {fid}", findings.get(fid) is state, f"got {actual}")
    for fid in expected["must_be_unknown"]:
        check(f"must_be_unknown: {fid}", fid not in findings, "finding was recorded")

    expectations = case["expectations"]
    rows = [row for panel in PANELS for row in payload[panel]]
    for condition in expectations["must_appear_on_cant_miss"]:
        check(f"must_appear_on_cant_miss: {condition}", has_condition(payload["cant_miss"], condition, engine))
    for condition in expectations["must_appear_somewhere"]:
        check(f"must_appear_somewhere: {condition}", has_condition(rows, condition, engine))

    question = payload.get("next_question")
    next_id = question["finding"] if question else None
    allowed = expectations["next_question_in"]
    check("next_question_in", next_id in allowed if allowed else next_id is None, f"got {next_id!r}; expected {allowed!r}")
    if expectations.get("no_number_in_output", True):
        issues = number_issues(payload)
        check("no_number_in_output", not issues, ", ".join(issues))
    if expectations.get("every_candidate_has_reasoning", False):
        check("every_candidate_has_reasoning", all(row.get("reasoning", "").strip() for row in rows))
    if expectations.get("cant_miss_entries_are_excludable", False):
        check("cant_miss_entries_are_excludable", all(row.get("exclusion_options") for row in payload["cant_miss"]))

    passed = all(results)
    print(f"{'PASS' if passed else 'FAIL'} {case['case']['id']}: {sum(results)}/{len(results)} expectations passed")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id", nargs="?", help="Run only this case.id (default: all cases)")
    parser.add_argument("--extract-model", help="Override the extraction model for this live run")
    parser.add_argument("--propose-model", help="Override the proposal model for this live run")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    for task in ("extract", "propose"):
        if getattr(args, f"{task}_model"):
            os.environ[f"SARVAM_{task.upper()}_MODEL"] = getattr(args, f"{task}_model")
    if not os.environ.get("SARVAM_API_KEY", "").strip():
        print("SARVAM_API_KEY is missing. Set it in .env or the environment before running live cases.", file=sys.stderr)
        return 2

    # Provider modules are loaded only after the explicit credential check.
    import extract
    import propose
    import llm
    from engine import Engine

    try:
        cases = load_cases()
        if args.case_id:
            selected = [case for case in cases if case["case"]["id"] == args.case_id]
            if not selected:
                available = ", ".join(case["case"]["id"] for case in cases)
                raise ValueError(f"Unknown case ID {args.case_id!r}. Available: {available}")
            cases = selected
        engine = Engine.load()
        print(f"Models: extract={llm.config('extract')['model']}, propose={llm.config('propose')['model']}")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Cannot load cases: {exc}", file=sys.stderr)
        return 2

    passed = 0
    for case in cases:
        case_id = case["case"]["id"]
        print(f"\nCase: {case_id} — {case['case'].get('summary', '')}")
        try:
            # Same extract -> propose -> triage sequence as main.py /analyze.
            findings, quotes = extract.extract(case["input"]["transcript"], case["input"]["document_text"])
            candidates = propose.propose(findings)
            payload = engine.assess(candidates, findings, confirmed_exclusions=(), asked=())
            payload["_findings"] = findings
            payload["_quotes"] = quotes
            passed += check_expectations(case, findings, payload, engine)
        except Exception as exc:
            print(f"FAIL {case_id}: pipeline/check error ({type(exc).__name__}): {exc}")
    print(f"\n{passed}/{len(cases)} cases passed. These are case regressions, not measured clinical accuracy.")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
