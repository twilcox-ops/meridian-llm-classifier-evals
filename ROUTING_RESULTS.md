# Routing Results

Three-way breakdown from `python -m src.eval_harness` over the 152-record
iteration set: cheap-only (Haiku alone), expensive-only (Sonnet 5 alone),
and routed (Haiku, escalating to Sonnet 5 when Haiku's self-reported
confidence is not "high", via `route_classify()`).

| Leg | Full accuracy | Cost / 1,000 | p95 latency |
|---|---|---|---|
| cheap-only (Haiku) | 66.4% | $1.34 | 1315ms |
| expensive-only (Sonnet 5) | 68.4% | $2.94 | 3756ms |
| routed | 65.8% | $1.57 | 3612ms |

**Routing on Haiku's self-reported confidence did not improve accuracy
here.** Full accuracy went slightly *down* (66.4% → 65.8%, a difference of
one record out of 152), while cost rose ~18% ($1.34 → $1.57 per 1,000) and
p95 latency roughly tripled (1315ms → 3612ms) from the sequential
Haiku-then-Sonnet round trip on escalated records.

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
