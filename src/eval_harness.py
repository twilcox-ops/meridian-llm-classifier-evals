"""Single-command eval harness for the ticket classifier.

Runs classify() over data/iteration.jsonl and reports overall accuracy,
per-category accuracy, confusion matrices, self-reported-confidence
calibration, and cost/latency based on actual token usage and wall-clock
time. Also runs the spec's three-way cost/latency/accuracy comparison:
cheap-only (Haiku alone), expensive-only (Sonnet alone), and routed
(Haiku, escalating to Sonnet on confidence != "high" via route_classify()).

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

from src.classify import (
    CATEGORIES,
    CONFIDENCES,
    MODEL,
    SONNET_MODEL,
    URGENCIES,
    classify,
    classify_with_sonnet,
    route_classify,
)

# Anthropic pricing, USD per million tokens. Update these if a model or its
# pricing changes.
HAIKU_PRICE_PER_MTOK_INPUT = 1.00
HAIKU_PRICE_PER_MTOK_OUTPUT = 5.00

# Sonnet 5's introductory rate, active as of today (2026-08-26) through
# 2026-08-31. Update these to the standard rate ($3.00/$15.00) once that
# window ends, or the reported routed/expensive-only costs will understate
# the real price.
SONNET_PRICE_PER_MTOK_INPUT = 2.00
SONNET_PRICE_PER_MTOK_OUTPUT = 10.00

# Back-compat aliases used by the existing cheap-only detailed report.
PRICE_PER_MTOK_INPUT = HAIKU_PRICE_PER_MTOK_INPUT
PRICE_PER_MTOK_OUTPUT = HAIKU_PRICE_PER_MTOK_OUTPUT


def token_cost(input_tokens: int, output_tokens: int, price_in: float, price_out: float) -> float:
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[index]


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
    full_correct_total = 0
    schema_failures = 0
    total_input_tokens = 0
    total_output_tokens = 0
    # (record, ClassificationResult, latency_ms) per record — kept so the
    # routed-mode evaluation can reuse these Haiku calls (via route_classify's
    # optional haiku_result param) instead of repeating them.
    per_record = []

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

    # Per self-reported confidence level: how many records landed at that
    # level, and of those, how many were fully correct (category AND
    # urgency), category-correct, and urgency-correct. This measures whether
    # the model's own confidence signal is trustworthy; the routed leg
    # (below, via route_classify) is what actually acts on it.
    per_confidence_total = defaultdict(int)
    per_confidence_full_correct = defaultdict(int)
    per_confidence_category_correct = defaultdict(int)
    per_confidence_urgency_correct = defaultdict(int)

    for record in records:
        true_category = record["label_category"]
        true_urgency = record["label_urgency"]
        per_category_total[true_category] += 1
        per_category_urgency_total[true_category] += 1

        start = time.perf_counter()
        result = classify(record["text"])
        latency_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency_ms)
        per_record.append((record, result, latency_ms))

        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        if not result.schema_valid:
            schema_failures += 1
            continue

        confusion[true_category][result.category] += 1
        urgency_confusion[true_urgency][result.urgency] += 1

        category_correct = result.category == true_category
        urgency_correct = result.urgency == true_urgency

        if category_correct:
            category_correct_total += 1
            per_category_correct[true_category] += 1
        if urgency_correct:
            urgency_correct_total += 1
            per_category_urgency_correct[true_category] += 1
        if category_correct and urgency_correct:
            full_correct_total += 1

        per_confidence_total[result.confidence] += 1
        if category_correct:
            per_confidence_category_correct[result.confidence] += 1
        if urgency_correct:
            per_confidence_urgency_correct[result.confidence] += 1
        if category_correct and urgency_correct:
            per_confidence_full_correct[result.confidence] += 1

    total_cost_usd = token_cost(
        total_input_tokens, total_output_tokens, HAIKU_PRICE_PER_MTOK_INPUT, HAIKU_PRICE_PER_MTOK_OUTPUT
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

    p50_latency_ms = percentile(latencies_ms, 0.50)
    p95_latency_ms = percentile(latencies_ms, 0.95)

    confidence_breakdown = {
        level: {
            "count": per_confidence_total[level],
            "full_accuracy": (
                per_confidence_full_correct[level] / per_confidence_total[level]
                if per_confidence_total[level] else 0.0
            ),
            "category_accuracy": (
                per_confidence_category_correct[level] / per_confidence_total[level]
                if per_confidence_total[level] else 0.0
            ),
            "urgency_accuracy": (
                per_confidence_urgency_correct[level] / per_confidence_total[level]
                if per_confidence_total[level] else 0.0
            ),
        }
        for level in CONFIDENCES
        if per_confidence_total[level]
    }

    return {
        "model": MODEL,
        "total": total,
        "schema_failures": schema_failures,
        "category_accuracy": category_correct_total / total if total else 0.0,
        "urgency_accuracy": urgency_correct_total / total if total else 0.0,
        "full_accuracy": full_correct_total / total if total else 0.0,
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
        "confidence_breakdown": confidence_breakdown,
        "per_record": per_record,
    }


def summarize_cheap(results: dict) -> dict:
    """Build the cheap-only leg of the three-way breakdown directly from the
    detailed run_eval() results — no extra API calls, since it's the same
    Haiku-only pass already performed for the full report above.
    """
    return {
        "label": "cheap-only (Haiku)",
        "total": results["total"],
        "schema_failures": results["schema_failures"],
        "category_accuracy": results["category_accuracy"],
        "urgency_accuracy": results["urgency_accuracy"],
        "full_accuracy": results["full_accuracy"],
        "escalated_count": 0,
        "escalation_rate": 0.0,
        "total_cost_usd": results["total_cost_usd"],
        "cost_per_1000_usd": results["cost_per_1000_usd"],
        "p50_latency_ms": results["p50_latency_ms"],
        "p95_latency_ms": results["p95_latency_ms"],
    }


def run_expensive_eval(records: list[dict]) -> dict:
    """Expensive-only leg: every record classified fresh by Sonnet alone,
    no Haiku involved. This is the one leg that can't reuse cached calls —
    it's a genuinely separate 152-call run.
    """
    total = len(records)
    category_correct = 0
    urgency_correct = 0
    full_correct = 0
    schema_failures = 0
    total_input_tokens = 0
    total_output_tokens = 0
    latencies_ms = []

    for record in records:
        start = time.perf_counter()
        result = classify_with_sonnet(record["text"])
        latencies_ms.append((time.perf_counter() - start) * 1000)

        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        if not result.schema_valid:
            schema_failures += 1
            continue

        cat_ok = result.category == record["label_category"]
        urg_ok = result.urgency == record["label_urgency"]
        if cat_ok:
            category_correct += 1
        if urg_ok:
            urgency_correct += 1
        if cat_ok and urg_ok:
            full_correct += 1

    total_cost_usd = token_cost(
        total_input_tokens, total_output_tokens, SONNET_PRICE_PER_MTOK_INPUT, SONNET_PRICE_PER_MTOK_OUTPUT
    )

    return {
        "label": f"expensive-only ({SONNET_MODEL})",
        "total": total,
        "schema_failures": schema_failures,
        "category_accuracy": category_correct / total if total else 0.0,
        "urgency_accuracy": urgency_correct / total if total else 0.0,
        "full_accuracy": full_correct / total if total else 0.0,
        "escalated_count": 0,
        "escalation_rate": 0.0,
        "total_cost_usd": total_cost_usd,
        "cost_per_1000_usd": (total_cost_usd / total) * 1000 if total else 0.0,
        "p50_latency_ms": percentile(latencies_ms, 0.50),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
    }


def run_routed_eval(per_record_haiku: list[tuple]) -> dict:
    """Routed leg: reuses the cached (record, haiku_result, haiku_latency_ms)
    triples from the cheap-only pass via route_classify()'s optional
    haiku_result param, so records that stay at confidence="high" cost no
    extra API call — only the ~confidence!=high subset makes a fresh Sonnet
    call here.
    """
    total = len(per_record_haiku)
    category_correct = 0
    urgency_correct = 0
    full_correct = 0
    schema_failures = 0
    escalated_count = 0
    total_cost_usd = 0.0
    latencies_ms = []

    for record, haiku_result, haiku_latency_ms in per_record_haiku:
        start = time.perf_counter()
        routed_result = route_classify(record["text"], haiku_result=haiku_result)
        call_latency_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(haiku_latency_ms + call_latency_ms)

        if routed_result.escalated:
            escalated_count += 1

        total_cost_usd += token_cost(
            routed_result.haiku_input_tokens,
            routed_result.haiku_output_tokens,
            HAIKU_PRICE_PER_MTOK_INPUT,
            HAIKU_PRICE_PER_MTOK_OUTPUT,
        )
        total_cost_usd += token_cost(
            routed_result.sonnet_input_tokens,
            routed_result.sonnet_output_tokens,
            SONNET_PRICE_PER_MTOK_INPUT,
            SONNET_PRICE_PER_MTOK_OUTPUT,
        )

        if not routed_result.schema_valid:
            schema_failures += 1
            continue

        cat_ok = routed_result.category == record["label_category"]
        urg_ok = routed_result.urgency == record["label_urgency"]
        if cat_ok:
            category_correct += 1
        if urg_ok:
            urgency_correct += 1
        if cat_ok and urg_ok:
            full_correct += 1

    return {
        "label": "routed (escalate on confidence != high)",
        "total": total,
        "schema_failures": schema_failures,
        "category_accuracy": category_correct / total if total else 0.0,
        "urgency_accuracy": urgency_correct / total if total else 0.0,
        "full_accuracy": full_correct / total if total else 0.0,
        "escalated_count": escalated_count,
        "escalation_rate": escalated_count / total if total else 0.0,
        "total_cost_usd": total_cost_usd,
        "cost_per_1000_usd": (total_cost_usd / total) * 1000 if total else 0.0,
        "p50_latency_ms": percentile(latencies_ms, 0.50),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
    }


def print_three_way(summaries: list[dict]) -> None:
    print("=" * 72)
    print("THREE-WAY BREAKDOWN: cheap-only / expensive-only / routed")
    print("=" * 72)
    for s in summaries:
        print(f"\n{s['label']}")
        print(
            f"  category={s['category_accuracy']:.1%}  urgency={s['urgency_accuracy']:.1%}  "
            f"full={s['full_accuracy']:.1%}  schema_failures={s['schema_failures']}/{s['total']}"
        )
        if s["escalated_count"]:
            print(
                f"  escalated to {SONNET_MODEL}: {s['escalated_count']}/{s['total']} "
                f"({s['escalation_rate']:.1%})"
            )
        print(
            f"  cost: ${s['total_cost_usd']:.4f} total, ${s['cost_per_1000_usd']:.4f} per 1,000"
        )
        print(f"  latency: p50={s['p50_latency_ms']:.0f}ms p95={s['p95_latency_ms']:.0f}ms")


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
    print("Self-reported confidence breakdown (cheap-only pass; see routed leg below for what's acted on):")
    for level in CONFIDENCES:
        stats = results["confidence_breakdown"].get(level)
        if stats is None:
            print(f"  {level:<8} 0 records")
            continue
        print(
            f"  {level:<8} {stats['count']:>3} records  "
            f"full={stats['full_accuracy']:.1%}  "
            f"category={stats['category_accuracy']:.1%}  "
            f"urgency={stats['urgency_accuracy']:.1%}"
        )
    print()
    # This is the cheap-only (Haiku) leg's detail. The full three-way
    # breakdown (cheap-only / expensive-only / routed) prints separately —
    # see print_three_way() in main().
    print(
        f"Tokens: {results['total_input_tokens']} in / "
        f"{results['total_output_tokens']} out"
    )
    print("Cost and latency (cheap-only leg; see three-way breakdown below):")
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

    print()
    cheap_summary = summarize_cheap(results)
    expensive_summary = run_expensive_eval(records)
    routed_summary = run_routed_eval(results["per_record"])
    print_three_way([cheap_summary, expensive_summary, routed_summary])


if __name__ == "__main__":
    main()
