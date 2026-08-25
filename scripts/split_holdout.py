"""Split support-tickets-eval.jsonl into a held-out set and an iteration set.

Stratifies on the `ambiguous` flag so both sets preserve the source's
~6.25% ambiguous-record ratio (12/192). Uses a fixed seed so the split
is reproducible — do not change SEED once iteration has started, or the
held-out set stops being a fixed, untouched reference.

The held-out 40 records must never be looked at during prompt iteration.
Only `data/iteration.jsonl` is used while tuning; `data/holdout.jsonl` is
read once, at the end, for the final reported numbers.
"""

import json
import random
from pathlib import Path

SEED = 42
HOLDOUT_SIZE = 40

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

SOURCE_PATH = REPO_ROOT / "sample-data" / "support-tickets-eval.jsonl"
DATA_DIR = PROJECT_ROOT / "data"
HOLDOUT_PATH = DATA_DIR / "holdout.jsonl"
ITERATION_PATH = DATA_DIR / "iteration.jsonl"


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def stratified_holdout_count(stratum_size: int, total_size: int, holdout_size: int) -> int:
    """Proportional allocation, rounded down; remainder resolved by caller."""
    return (stratum_size * holdout_size) // total_size


def split(records: list[dict], seed: int, holdout_size: int) -> tuple[list[dict], list[dict]]:
    ambiguous = [r for r in records if r.get("ambiguous") is True]
    normal = [r for r in records if r.get("ambiguous") is not True]
    total = len(records)

    # Largest-remainder method: floor each stratum's proportional share,
    # then hand out the leftover slots to the stratum(s) with the largest
    # fractional remainder, so the two strata's holdout counts sum to
    # exactly `holdout_size` while staying as proportional as integers allow.
    raw_ambiguous = len(ambiguous) * holdout_size / total
    raw_normal = len(normal) * holdout_size / total
    floor_ambiguous = int(raw_ambiguous)
    floor_normal = int(raw_normal)
    remainder = holdout_size - (floor_ambiguous + floor_normal)

    remainders = sorted(
        [("ambiguous", raw_ambiguous - floor_ambiguous), ("normal", raw_normal - floor_normal)],
        key=lambda pair: pair[1],
        reverse=True,
    )
    counts = {"ambiguous": floor_ambiguous, "normal": floor_normal}
    for name, _ in remainders[:remainder]:
        counts[name] += 1

    rng = random.Random(seed)
    ambiguous_shuffled = ambiguous[:]
    normal_shuffled = normal[:]
    rng.shuffle(ambiguous_shuffled)
    rng.shuffle(normal_shuffled)

    holdout = ambiguous_shuffled[: counts["ambiguous"]] + normal_shuffled[: counts["normal"]]
    iteration = ambiguous_shuffled[counts["ambiguous"] :] + normal_shuffled[counts["normal"] :]

    # Shuffle each output set's record order (not just stratum order) so
    # ambiguous records aren't clustered at the front of either file.
    rng.shuffle(holdout)
    rng.shuffle(iteration)

    return holdout, iteration


def main() -> None:
    records = load_records(SOURCE_PATH)
    holdout, iteration = split(records, SEED, HOLDOUT_SIZE)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(HOLDOUT_PATH, holdout)
    write_jsonl(ITERATION_PATH, iteration)

    def ambiguous_count(rs: list[dict]) -> int:
        return sum(1 for r in rs if r.get("ambiguous") is True)

    print(f"Source records: {len(records)}")
    print(f"Held-out:  {len(holdout)} records ({ambiguous_count(holdout)} ambiguous) -> {HOLDOUT_PATH}")
    print(f"Iteration: {len(iteration)} records ({ambiguous_count(iteration)} ambiguous) -> {ITERATION_PATH}")


if __name__ == "__main__":
    main()
