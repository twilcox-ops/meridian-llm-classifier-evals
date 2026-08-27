"""Final held-out evaluation — the one and only time data/holdout.jsonl is
used for scoring the classifier.

Runs the exact same v2 classify() logic already measured against
data/iteration.jsonl (see PROMPT_VERSIONS.md and README.md's Measurements
section) against the 40 records set aside by scripts/split_holdout.py and
never looked at during any prompt iteration. Reuses run_eval() and
print_report() from src.eval_harness completely unchanged, so this is the
identical measurement and report format as the iteration-set runs above —
just against data the classifier was never tuned on.

Run this exactly once, at the end, after prompt iteration is finished.
Re-running it against holdout data — or worse, tuning against a result it
produces — defeats the entire point of holding it out in the first place.

Usage: python -m scripts.run_holdout
"""

from pathlib import Path

from src.eval_harness import load_records, print_report, run_eval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DATA_PATH = PROJECT_ROOT / "data" / "holdout.jsonl"


def main() -> None:
    print("=" * 72)
    print("FINAL HELD-OUT EVALUATION -- data/holdout.jsonl, 40 records")
    print("This is the one and only scoring run against the held-out set.")
    print("=" * 72)
    print()

    records = load_records(HOLDOUT_DATA_PATH)
    results = run_eval(records)
    print_report(results)


if __name__ == "__main__":
    main()
