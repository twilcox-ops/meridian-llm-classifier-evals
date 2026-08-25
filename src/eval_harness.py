"""Single-command eval harness for the ticket classifier.

Runs classify() over data/iteration.jsonl and reports overall accuracy,
per-category accuracy, and cost based on actual token usage.

NEVER point this at data/holdout.jsonl. The held-out 40 records are read
exactly once, at the end of iteration, for the final reported numbers —
running this harness against them during tuning would make that number
meaningless.

Usage: python -m src.eval_harness
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.classify import CATEGORIES, MODEL, URGENCIES, classify

# Anthropic pricing, USD per million tokens, for the pinned MODEL above.
# Update these if the model or its pricing changes.
PRICE_PER_MTOK_INPUT = 1.00
PRICE_PER_MTOK_OUTPUT = 5.00

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ITERATION_DATA_PATH = PROJECT_ROOT / "data" / "iteration.jsonl"


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_eval(records: list[dict]) -> dict:
    total = len(records)
    category_correct_total = 0
    urgency_correct_total = 0
    schema_failures = 0
    total_input_tokens = 0
    total_output_tokens = 0

    per_category_total = defaultdict(int)
    per_category_correct = defaultdict(int)
    per_category_urgency_total = defaultdict(int)
    per_category_urgency_correct = defaultdict(int)

    # confusion[actual][predicted] = count. Only populated for schema-valid
    # responses — a schema failure has no predicted category to place in
    # the matrix, and is already counted separately via schema_failures.
    confusion = {actual: defaultdict(int) for actual in CATEGORIES}
    urgency_confusion = {actual: defaultdict(int) for actual in URGENCIES}
    latencies_ms = []

    for record in records:
        true_category = record["label_category"]
        true_urgency = record["label_urgency"]
        per_category_total[true_category] += 1
        per_category_urgency_total[true_category] += 1

        start = time.perf_counter()
        result = classify(record["text"])
        latencies_ms.append((time.perf_counter() - start) * 1000)

        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        if not result.schema_valid:
            schema_failures += 1
            continue

        confusion[true_category][result.category] += 1
        urgency_confusion[true_urgency][result.urgency] += 1

        if result.category == true_category:
            category_correct_total += 1
            per_category_correct[true_category] += 1
        if result.urgency == true_urgency:
            urgency_correct_total += 1
            per_category_urgency_correct[true_category] += 1

    total_cost_usd = (
        (total_input_tokens / 1_000_000) * PRICE_PER_MTOK_INPUT
        + (total_output_tokens / 1_000_000) * PRICE_PER_MTOK_OUTPUT
    )
    cost_per_1000_usd = (total_cost_usd / total) * 1000 if total else 0.0

    per_category_accuracy = {
        cat: per_category_correct[cat] / per_category_total[cat]
        for cat in per_category_total
    }
    per_category_urgency_accuracy = {
        cat: per_category_urgency_correct[cat] / per_category_urgency_total[cat]
        for cat in per_category_urgency_total
    }

    def largest_off_diagonal(matrix: dict) -> tuple[Optional[tuple], int]:
        largest = None
        largest_count = 0
        for actual, predicted_counts in matrix.items():
            for predicted, count in predicted_counts.items():
                if predicted != actual and count > largest_count:
                    largest = (actual, predicted)
                    largest_count = count
        return largest, largest_count

    largest_confusion, largest_confusion_count = largest_off_diagonal(confusion)
    largest_urgency_confusion, largest_urgency_confusion_count = largest_off_diagonal(
        urgency_confusion
    )

    def percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(int(len(ordered) * pct), len(ordered) - 1)
        return ordered[index]

    p50_latency_ms = percentile(latencies_ms, 0.50)
    p95_latency_ms = percentile(latencies_ms, 0.95)

    return {
        "model": MODEL,
        "total": total,
        "schema_failures": schema_failures,
        "category_accuracy": category_correct_total / total if total else 0.0,
        "urgency_accuracy": urgency_correct_total / total if total else 0.0,
        "per_category_accuracy": per_category_accuracy,
        "per_category_urgency_accuracy": per_category_urgency_accuracy,
        "confusion": confusion,
        "largest_confusion": largest_confusion,
        "largest_confusion_count": largest_confusion_count,
        "urgency_confusion": urgency_confusion,
        "largest_urgency_confusion": largest_urgency_confusion,
        "largest_urgency_confusion_count": largest_urgency_confusion_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": total_cost_usd,
        "cost_per_1000_usd": cost_per_1000_usd,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
    }


def print_matrix(title: str, matrix: dict, labels: list[str]) -> None:
    print(title)
    header = "actual\\predicted".ljust(16) + "".join(label[:10].ljust(11) for label in labels)
    print(header)
    for actual in labels:
        row = matrix[actual]
        line = actual.ljust(16) + "".join(str(row.get(pred, 0)).ljust(11) for pred in labels)
        print(line)


def print_largest_confusion(kind: str, largest: Optional[tuple], count: int) -> None:
    if largest is None:
        print(f"Largest {kind} confusion: none - predictions matched actual for every record.")
    else:
        actual, predicted = largest
        print(f"Largest {kind} confusion: {count}x actual={actual} predicted as {predicted}")


def print_report(results: dict) -> None:
    print(f"Model: {results['model']}")
    print(f"Records evaluated: {results['total']}")
    print(f"Schema-validation failures: {results['schema_failures']}")
    print()
    print(f"Overall category accuracy: {results['category_accuracy']:.1%}")
    print(f"Overall urgency accuracy:  {results['urgency_accuracy']:.1%}")
    print()
    print("Per-category accuracy:")
    for cat, acc in sorted(results["per_category_accuracy"].items()):
        print(f"  {cat:<12} {acc:.1%}")
    print()
    print("Per-category urgency accuracy:")
    for cat, acc in sorted(results["per_category_urgency_accuracy"].items()):
        print(f"  {cat:<12} {acc:.1%}")
    print()
    print_matrix("Category confusion matrix (rows=actual, cols=predicted):", results["confusion"], CATEGORIES)
    print_largest_confusion(
        "category", results["largest_confusion"], results["largest_confusion_count"]
    )
    print()
    print_matrix(
        "Urgency confusion matrix (rows=actual, cols=predicted):",
        results["urgency_confusion"],
        URGENCIES,
    )
    print_largest_confusion(
        "urgency", results["largest_urgency_confusion"], results["largest_urgency_confusion_count"]
    )
    print()
    # Only one model is called right now, so this is the "cheap-only" leg of
    # the spec's three-way cost/latency breakdown (cheap-only / expensive-only
    # / routed). The other two legs need confidence-based routing added
    # first — not implemented yet.
    print(
        f"Tokens: {results['total_input_tokens']} in / "
        f"{results['total_output_tokens']} out"
    )
    print("Cost and latency (cheap-only - single model, no routing yet):")
    print(
        f"  Cost: ${results['total_cost_usd']:.4f} total, "
        f"${results['cost_per_1000_usd']:.4f} per 1,000 classifications"
    )
    print(
        f"  Latency: p50={results['p50_latency_ms']:.0f}ms "
        f"p95={results['p95_latency_ms']:.0f}ms"
    )


def main() -> None:
    records = load_records(ITERATION_DATA_PATH)
    results = run_eval(records)
    print_report(results)


if __name__ == "__main__":
    main()
