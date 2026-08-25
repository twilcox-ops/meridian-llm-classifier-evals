"""Ticket classification via Claude's tool-use / structured-output mode.

Returns a typed (category, urgency) pair for one ticket's text. Never parses
prose with regex: the model is forced to call a single tool with an enum-
constrained JSON schema, and a response that doesn't validate against that
schema is recorded as a failure (schema_valid=False), not silently repaired.
"""

from dataclasses import dataclass
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Pinned to an exact, dated snapshot (not a floating alias) so the eval
# harness is reproducible: re-running later calls the same model behavior.
# Do not change this without re-running the full eval and updating the
# prompt-version table.
MODEL = "claude-haiku-4-5-20251001"

CATEGORIES = ["billing", "scheduling", "outage", "compliance", "sales"]
URGENCIES = ["low", "medium", "high", "critical"]

# v2 system prompt. Adds explicit per-category urgency rules derived by
# cross-tabbing category x urgency over data/iteration.jsonl (the training
# set) — not guessed. That cross-tab showed urgency is almost entirely a
# function of category: compliance is always medium; outage is only
# high/critical, split cleanly by person-in-danger + emergency-response
# language; sales/scheduling/billing are only low/medium, and that low/
# medium split carries no recoverable textual signal (byte-identical ticket
# text appeared under both labels), so no rule is stated for it here.
SYSTEM_PROMPT = (
    "You are a support-ticket triage classifier for a building-equipment "
    "service company. Classify each ticket into exactly one category and "
    "one urgency level.\n\n"
    "Tie-break rule: when a ticket plausibly spans two categories, a "
    "safety-affecting issue (e.g. a trapped passenger, a stuck or "
    "malfunctioning unit, a code/compliance violation) outranks a purely "
    "commercial one (billing, sales) — classify by the safety-affecting "
    "aspect.\n\n"
    "Urgency rules, by category:\n"
    "- compliance tickets are always urgency=medium.\n"
    "- outage tickets are only urgency=high or urgency=critical, never "
    "low or medium. Use critical only when there is an explicit "
    "person-in-danger plus emergency-response signal (e.g. a trapped "
    "passenger, a call to 911/emergency services). Otherwise use high.\n"
    "- sales, scheduling, and billing tickets are only urgency=low or "
    "urgency=medium, never high or critical."
)

CLASSIFY_TOOL = {
    "name": "classify_ticket",
    "description": "Record the category and urgency classification for one support ticket.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": CATEGORIES},
            "urgency": {"type": "string", "enum": URGENCIES},
        },
        "required": ["category", "urgency"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class ClassificationResult:
    category: Optional[str]
    urgency: Optional[str]
    input_tokens: int
    output_tokens: int
    schema_valid: bool
    error: Optional[str] = None


_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def classify(ticket_text: str) -> ClassificationResult:
    """Classify one ticket's text. Never raises for a model-side failure —
    API errors and schema failures both come back as a ClassificationResult
    with schema_valid=False, so the eval harness can count them as a
    recorded failure mode rather than crashing the run.
    """
    client = _get_client()

    try:
        # temperature isn't in this SDK's typed messages.create() signature
        # (anthropic 1.x dropped temperature/top_p/top_k from the method for
        # every model), but the API itself still accepts temperature for
        # this model — confirmed directly: a raw call with
        # extra_body={"temperature": 0} against claude-haiku-4-5-20251001
        # returns normally, no 400. So it goes through extra_body instead of
        # as a named kwarg. Not a determinism gap.
        response = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            tools=[CLASSIFY_TOOL],
            tool_choice={"type": "tool", "name": "classify_ticket"},
            extra_body={"temperature": 0},
            messages=[{"role": "user", "content": ticket_text}],
        )
    except anthropic.APIError as e:
        return ClassificationResult(
            category=None,
            urgency=None,
            input_tokens=0,
            output_tokens=0,
            schema_valid=False,
            error=f"API error: {e}",
        )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        return ClassificationResult(
            category=None,
            urgency=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            schema_valid=False,
            error="no tool_use block in response",
        )

    payload = tool_use_block.input
    category = payload.get("category")
    urgency = payload.get("urgency")

    if category not in CATEGORIES or urgency not in URGENCIES:
        return ClassificationResult(
            category=category,
            urgency=urgency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            schema_valid=False,
            error=f"value outside enum: category={category!r} urgency={urgency!r}",
        )

    return ClassificationResult(
        category=category,
        urgency=urgency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        schema_valid=True,
    )
