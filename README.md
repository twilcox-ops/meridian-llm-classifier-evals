# Project 4 — LLM Classifier With an Eval Harness

## Problem

A support-ticket classifier for a building-equipment service company (think
elevator/facilities maintenance): given raw ticket text, it assigns one of
five categories (`billing`, `scheduling`, `outage`, `compliance`, `sales`)
and one of four urgency levels (`low`, `medium`, `high`, `critical`).

Calling an LLM to do this is the easy part. The hard part — and the actual
point of this project — is knowing how well it works: an accuracy number
measured against a held-out set the model never saw during tuning, a
confusion matrix that shows *where* it's wrong, and a record of whether a
prompt change made things better or worse. Anyone can get a plausible
answer out of a model; the eval harness is what turns that into something
you can defend and regression-test.

## Architecture

```mermaid
flowchart LR
    Ticket["Ticket text"] --> Classify["classify()<br/>Haiku 4.5, structured tool-use,<br/>schema-validated, temperature 0"]
    Classify -->|confidence == high| Result["category + urgency"]
    Classify -->|confidence != high<br/>or schema failure| Escalate["classify_with_sonnet()<br/>Sonnet 5 re-classification"]
    Escalate --> Result
```

`route_classify()` wraps this decision: it calls the cheap model (Haiku)
first, and only calls the expensive model (Sonnet) when Haiku's own
self-reported confidence isn't `"high"` — including the case where Haiku's
response fails schema validation outright.

Where things live:

- **`src/classify.py`** — the classification core: the tool schema, system
  prompt, `classify()` (Haiku), `classify_with_sonnet()` (Sonnet), and
  `route_classify()` (the routing decision between them).
- **`src/eval_harness.py`** — the single-command eval harness
  (`python -m src.eval_harness`): runs the iteration set through all three
  modes (cheap-only, expensive-only, routed) and reports accuracy, a
  confusion matrix, cost, and latency for each.
- **`scripts/split_holdout.py`** — one-time script that splits the labeled
  data into the held-out and iteration sets, stratified so both keep the
  source data's ambiguous-ticket ratio.
- **`scripts/test_injection.py`** — runs a small set of synthetic tickets
  containing embedded prompt-injection attempts through `classify()` and
  reports whether each one was resisted.

## Tie-break rule

The source data includes tickets that genuinely span two categories — a
unit outage that also involves a billing dispute ("unit is down AND we
were billed anyway"), or a code violation that also touches a commercial
account issue. Forcing a single category onto these means picking a side,
and that pick needs to be a stated rule, not an implicit habit the model
falls into differently on different days.

The rule, encoded directly in `SYSTEM_PROMPT` in `src/classify.py`:
**a safety-affecting issue (outage, compliance) outranks a purely
commercial one (billing, sales).** A ticket describing both a stuck
elevator and a billing error is classified `outage`, not `billing`.

Why safety wins: a stark asymmetry between the two failure modes.
Misclassifying a safety issue as a commercial one means it can queue behind
routine paperwork, get triaged by someone without the authority or urgency
to act on it, or simply sit — and the downstream cost of a missed or
delayed safety response (liability, injury, an actual emergency going
unaddressed) is categorically worse than the cost of a billing question
waiting an extra day. Misclassifying a commercial issue as a safety one is
the safe direction to err in: worst case, it gets looked at sooner than it
needed to be. There's no such thing as "too fast" on a real hazard, but
there is such a thing as "too slow."

This rule was written into the system prompt before the classifier was run
against any data — it's a decision made in advance, not a patch added
after seeing which tickets the model got wrong. That ordering matters: a
tie-break rule invented to explain away an observed failure is post-hoc
rationalization, while one stated up front and then measured against is an
actual, falsifiable design decision.
