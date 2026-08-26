"""Ticket classification via Claude's tool-use / structured-output mode.

Returns a typed (category, urgency, confidence) triple for one ticket's
text. Never parses prose with regex: the model is forced to call a single
tool with an enum-constrained JSON schema, and a response that doesn't
validate against that schema is recorded as a failure (schema_valid=False),
not silently repaired.

Also provides route_classify(): confidence-based routing that escalates
to a larger model (Sonnet 5) only when the cheap model (Haiku 4.5) isn't
confident, or fails schema validation.
"""

from dataclasses import dataclass
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Pinned to an exact, dated snapshot (not a floating alias) so the eval
# harness is reproducible: re-running later calls the same model behavior.
# Do not change this without re-running the full eval and updating the
# prompt-version table. This is the "cheap" model in the routing scheme.
MODEL = "claude-haiku-4-5-20251001"

# The "expensive" escalation model for routing. Current-generation model —
# confirmed it rejects `temperature` outright (400: "temperature is
# deprecated for this model"), unlike Haiku 4.5 where the API silently
# accepts it via extra_body. So calls to this model omit temperature
# entirely rather than passing 0.
SONNET_MODEL = "claude-sonnet-5"

CATEGORIES = ["billing", "scheduling", "outage", "compliance", "sales"]
URGENCIES = ["low", "medium", "high", "critical"]
CONFIDENCES = ["low", "medium", "high"]

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
            "confidence": {
                "type": "string",
                "enum": CONFIDENCES,
                "description": (
                    "Your confidence that both category and urgency above "
                    "are correct."
                ),
            },
        },
        "required": ["category", "urgency", "confidence"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class ClassificationResult:
    category: Optional[str]
    urgency: Optional[str]
    confidence: Optional[str]
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


def _classify_with_model(ticket_text: str, model: str, use_temperature: bool) -> ClassificationResult:
    """Shared request/parse logic for both classify() and the Sonnet
    escalation call. use_temperature controls whether extra_body carries
    temperature=0 — Haiku 4.5 accepts it, current-gen models like Sonnet 5
    reject it outright (400), so callers must pass False for those.
    """
    client = _get_client()

    request_kwargs = dict(
        model=model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_ticket"},
        messages=[{"role": "user", "content": ticket_text}],
    )
    if use_temperature:
        request_kwargs["extra_body"] = {"temperature": 0}

    try:
        response = client.messages.create(**request_kwargs)
    except anthropic.APIError as e:
        return ClassificationResult(
            category=None,
            urgency=None,
            confidence=None,
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
            confidence=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            schema_valid=False,
            error="no tool_use block in response",
        )

    payload = tool_use_block.input
    category = payload.get("category")
    urgency = payload.get("urgency")
    confidence = payload.get("confidence")

    if category not in CATEGORIES or urgency not in URGENCIES or confidence not in CONFIDENCES:
        return ClassificationResult(
            category=category,
            urgency=urgency,
            confidence=confidence,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            schema_valid=False,
            error=(
                f"value outside enum: category={category!r} urgency={urgency!r} "
                f"confidence={confidence!r}"
            ),
        )

    return ClassificationResult(
        category=category,
        urgency=urgency,
        confidence=confidence,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        schema_valid=True,
    )


def classify(ticket_text: str) -> ClassificationResult:
    """Classify one ticket's text with the pinned cheap model (Haiku).
    Never raises for a model-side failure — API errors and schema failures
    both come back as a ClassificationResult with schema_valid=False, so
    callers can count them as a recorded failure mode rather than crashing.
    """
    return _classify_with_model(ticket_text, MODEL, use_temperature=True)


def classify_with_sonnet(ticket_text: str) -> ClassificationResult:
    """Classify one ticket's text with the escalation model (Sonnet 5)."""
    return _classify_with_model(ticket_text, SONNET_MODEL, use_temperature=False)


@dataclass
class RoutedClassificationResult:
    """Result of route_classify(): the final (category, urgency, confidence)
    plus enough per-model token detail to price the call accurately, since a
    routed call may have spent tokens on Haiku only, or on both Haiku and
    Sonnet.
    """

    category: Optional[str]
    urgency: Optional[str]
    confidence: Optional[str]
    schema_valid: bool
    escalated: bool
    haiku_input_tokens: int
    haiku_output_tokens: int
    sonnet_input_tokens: int
    sonnet_output_tokens: int
    error: Optional[str] = None

    @property
    def input_tokens(self) -> int:
        return self.haiku_input_tokens + self.sonnet_input_tokens

    @property
    def output_tokens(self) -> int:
        return self.haiku_output_tokens + self.sonnet_output_tokens


def route_classify(
    ticket_text: str, haiku_result: Optional[ClassificationResult] = None
) -> RoutedClassificationResult:
    """Classify with cheap-model-first routing: call Haiku, and escalate to
    Sonnet only when Haiku's self-reported confidence isn't "high" —
    covering both an explicit low/medium confidence and a schema failure
    (confidence is None), matching "escalate on low confidence or schema
    failure". Records at confidence="high" keep the Haiku result as-is.

    haiku_result lets a caller that already has a fresh Haiku result for
    this exact ticket (e.g. the eval harness's cheap-only pass) pass it in
    directly, so route_classify doesn't repeat that API call purely to
    re-derive a routing decision it can already answer. Called with just
    ticket_text, it computes the Haiku result itself.
    """
    if haiku_result is None:
        haiku_result = classify(ticket_text)

    if haiku_result.schema_valid and haiku_result.confidence == "high":
        return RoutedClassificationResult(
            category=haiku_result.category,
            urgency=haiku_result.urgency,
            confidence=haiku_result.confidence,
            schema_valid=True,
            escalated=False,
            haiku_input_tokens=haiku_result.input_tokens,
            haiku_output_tokens=haiku_result.output_tokens,
            sonnet_input_tokens=0,
            sonnet_output_tokens=0,
        )

    sonnet_result = classify_with_sonnet(ticket_text)
    return RoutedClassificationResult(
        category=sonnet_result.category,
        urgency=sonnet_result.urgency,
        confidence=sonnet_result.confidence,
        schema_valid=sonnet_result.schema_valid,
        escalated=True,
        haiku_input_tokens=haiku_result.input_tokens,
        haiku_output_tokens=haiku_result.output_tokens,
        sonnet_input_tokens=sonnet_result.input_tokens,
        sonnet_output_tokens=sonnet_result.output_tokens,
        error=sonnet_result.error,
    )
