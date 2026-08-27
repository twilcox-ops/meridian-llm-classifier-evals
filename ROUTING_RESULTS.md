# Routing Results

Three-way breakdown from `python -m src.eval_harness` over the 152-record
iteration set: cheap-only (Haiku alone), expensive-only (Sonnet 5 alone),
and routed (Haiku, escalating to Sonnet 5 when Haiku's self-reported
confidence is not "high", via `route_classify()`).

| Leg | Full accuracy | Cost / 1,000 | p95 latency |
|---|---|---|---|
| cheap-only (Haiku) | 66.4% | $1.33 | 1508ms |
| expensive-only (Sonnet 5) | 71.1% | $2.89 | 3153ms |
| routed | 66.4% | $1.58 | 3661ms |

**Note on reproducibility:** these three numbers move slightly between runs
of this harness — cheap-only's figures are pinned (Haiku, temperature 0)
and reproduce exactly, but expensive-only and routed both call Sonnet 5,
which rejects the `temperature` parameter outright (see the comment above
`classify_with_sonnet()` in `src/classify.py`), so those two legs are not
guaranteed bit-for-bit reproducible run to run. The table above is one
run's numbers, not a fixed constant — expect a few points of drift on
re-run.

**Routing on Haiku's self-reported confidence did not improve accuracy
here.** Full accuracy came back identical (66.4% cheap-only vs. 66.4%
routed — no net change), while cost rose ~19% ($1.33 → $1.58 per 1,000) and
p95 latency more than doubled (1508ms → 3661ms) from the sequential
Haiku-then-Sonnet round trip on the 13/152 (8.6%) escalated records. In an
earlier run, routed accuracy had actually landed a hair *below* cheap-only
(66.4% → 65.8%); this run it landed exactly even. Both outcomes tell the
same story: routing bought no measurable accuracy gain here, at real cost
and latency.

This is consistent with the earlier finding that the accuracy gap is a
**dataset label-noise ceiling, not a model-capability gap**: escalating to
a larger model doesn't reliably fix an urgency label that carries no
recoverable signal in the ticket text, and can even land on the wrong
answer where the cheap model happened to land on the right one (see the
12-record Haiku-vs-Sonnet spot check).

**Important scope caveat on this result:** the confidence signal only ever
flagged **scheduling** and **compliance** records for escalation — it never
flagged **billing** or **sales**, the two worst-performing categories on
urgency (35.5% and 53.3% respectively, vs. scheduling's 41.4%). So this run
does not test whether a stronger model would help billing or sales
specifically; it only tested escalation on the two categories the
confidence signal actually fired for. A separate test that force-escalates
billing/sales (rather than relying on Haiku's confidence to flag them)
would be needed to rule that out — not done here.
