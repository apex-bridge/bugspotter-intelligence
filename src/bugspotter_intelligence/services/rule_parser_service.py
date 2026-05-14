"""
Natural-language → structured DedupRule parser.

The flow is intentionally simple:
  1. Build a system prompt that defines the full ontology (triggers /
     conditions / actions) plus a handful of few-shot examples.
  2. Append the user's NL string and ask the LLM for strict JSON output.
  3. Try `json.loads`, then a regex fallback to pull JSON out of any markdown
     fences or pre/postamble the LLM may have wrapped around it.
  4. Validate the JSON against the `DedupRule` Pydantic model. Validation
     errors → return a null draft with a human-readable `errors` list rather
     than throwing.

Design choices:
- Low temperature (0.1) so the LLM sticks to the schema.
- Few-shot examples cover the three persona-driven cases (B1/B2/B3) plus one
  "ambiguous" case that demonstrates the `clarifications` escape hatch.
- The context (available integrations / Slack channels / templates) is
  injected into the prompt so the LLM doesn't hallucinate targets the tenant
  hasn't configured.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from ..llm.base import LLMProvider
from ..models.dedup_rule import DedupRule

logger = logging.getLogger(__name__)


# ============================================================================
# Prompt construction
# ============================================================================


_SYSTEM_PROMPT_BASE = """You convert natural-language descriptions of bug-tracking automation rules into structured JSON.

# Vocabulary

## Triggers (`when`) — pick exactly ONE
- `duplicate_detected`: a bug was just identified as a duplicate of an existing one
- `outbox_about_to_skip`: a duplicate is about to be silently suppressed (use this for "+1 counter", "auto-reopen" type rules)
- `cluster_growing` { threshold: int, window: "1h" | "24h" | etc }: a cluster grew quickly
- `schedule` { cron: "0 9 * * 1" }: recurring on cron schedule (use for digests)

## Conditions (`if`) — zero or more, ANDed
Each: { field, op, value, window? }

Fields:
- `canonical.status` (op: eq|in|not_in; value: list of [open, in_progress, closed, wont_fix])
- `canonical.closed_days_ago` (op: gte|lte; value: int)
- `hits_in_window` (op: gte; value: int; REQUIRES `window`)
- `reporter.customer.tier` (op: eq|in; value: "enterprise"|"free"|...)
- `severity` (op: eq|in; value: list of [low, medium, high, critical])

## Actions (`then`) — at least ONE
- `ticket.add_comment` { target: "canonical", body: string }
- `ticket.transition` { target: "canonical", to: "open"|"in_progress"|"closed"|"wont_fix" }
- `notify.email` { to: "reporter"|"closer"|"all_reporters"|<email>, template: string }
- `notify.slack` { channel: "#name" | user: "closer"|<handle>, message: string }
- `notify.webhook` { url, payload? }

Rate limit (optional, top-level): `rate_limit: { count: int, window: "1h"|"24h" }`.

# Output

Output ONE JSON object with this exact top-level shape:

{
  "draft": { "name": ..., "when": ..., "if": [...], "then": [...], "rate_limit": ..., "enabled": true } | null,
  "errors": ["..."],
  "clarifications": ["..."]
}

- If you can confidently parse: set `draft` and leave `errors`/`clarifications` empty.
- If a required parameter is missing (e.g. "ping me in Slack" but no channel): put a placeholder AND list it in `clarifications`.
- If the input doesn't map to any available action: set `draft: null` and explain in `errors`.
- DO NOT invent integrations that aren't in the tenant's available list.

# Examples

## Example 1
Input: "when a closed bug gets three hits in a day, reopen it and add a comment"
Output:
{
  "draft": {
    "name": "Auto-reopen on regression",
    "when": { "type": "outbox_about_to_skip" },
    "if": [
      { "field": "canonical.status", "op": "in", "value": ["closed", "wont_fix"] },
      { "field": "hits_in_window", "op": "gte", "value": 3, "window": "24h" }
    ],
    "then": [
      { "type": "ticket.transition", "target": "canonical", "to": "in_progress" },
      { "type": "ticket.add_comment", "target": "canonical", "body": "Auto-reopened: {hits_24h} new hits in 24h after close." }
    ],
    "enabled": true
  },
  "errors": [],
  "clarifications": []
}

## Example 2
Input: "send the reporter an email when their bug is grouped"
Output:
{
  "draft": {
    "name": "Notify reporter on dedup",
    "when": { "type": "duplicate_detected" },
    "then": [
      { "type": "notify.email", "to": "reporter", "template": "dedup_ack" }
    ],
    "enabled": true
  },
  "errors": [],
  "clarifications": []
}

## Example 3
Input: "every monday morning email me the top bugs of the week"
Output:
{
  "draft": {
    "name": "Weekly digest",
    "when": { "type": "schedule", "cron": "0 9 * * 1" },
    "then": [
      { "type": "notify.email", "to": "REPLACE_ME@example.com", "template": "weekly_digest" }
    ],
    "enabled": true
  },
  "errors": [],
  "clarifications": ["What email address should receive the digest? I put REPLACE_ME as a placeholder."]
}

## Example 4
Input: "make important things faster"
Output:
{
  "draft": null,
  "errors": ["Input is too abstract to map to an action."],
  "clarifications": [
    "Which event should trigger this? (new duplicate, cluster growing, scheduled, ...)",
    "What does 'faster' mean — raise priority, send an alert, transition status?"
  ]
}
"""


def _build_tenant_context_block(
    available_integrations: list[str],
    available_slack_channels: list[str],
    available_email_templates: list[str],
) -> str:
    """Tell the LLM what the tenant actually has configured.

    Without this the LLM will happily produce rules referencing Slack
    channels or email templates the tenant doesn't have — which then fail
    silently at execution time.
    """
    lines: list[str] = ["# Tenant context"]
    lines.append(
        "Available ticket integrations: "
        + (", ".join(available_integrations) if available_integrations else "(none)")
    )
    lines.append(
        "Available Slack channels: "
        + (", ".join(available_slack_channels) if available_slack_channels else "(none)")
    )
    lines.append(
        "Available email templates: "
        + (", ".join(available_email_templates) if available_email_templates else "(none)")
    )
    lines.append("")
    lines.append(
        "If the user mentions a target you don't see here, prefer a clarification "
        "over inventing one."
    )
    return "\n".join(lines)


def build_prompt(
    nl_input: str,
    available_integrations: list[str],
    available_slack_channels: list[str],
    available_email_templates: list[str],
) -> str:
    """Assemble the full LLM prompt for a single NL parse."""
    tenant_block = _build_tenant_context_block(
        available_integrations,
        available_slack_channels,
        available_email_templates,
    )
    return (
        f"{_SYSTEM_PROMPT_BASE}\n\n"
        f"{tenant_block}\n\n"
        "# Parse this input\n\n"
        f"Input: {nl_input.strip()}\n\n"
        "Output the JSON object. No prose, no markdown fences."
    )


# ============================================================================
# Output parsing
# ============================================================================


def _extract_top_level_json_objects(text: str) -> list[str]:
    """Return every top-level `{...}` substring in `text`, in order.

    Uses a brace counter — handles arbitrary nesting depth, unlike a regex
    with hardcoded recursion bounds. Strings and their escape sequences are
    tracked so braces inside `"..."` don't throw off the depth.

    "Top-level" here means: not nested inside another `{...}` at the
    outermost scope. So a response like `{"a": {"b": 1}}` returns one
    candidate, not two.
    """
    candidates: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
                    start = -1

    return candidates


def _extract_json_object(raw: str) -> dict | None:
    """Pull the first plausible JSON object out of an LLM response.

    Strategy:
      1. Try parsing the whole string.
      2. Try the contents of any ```json``` (or generic ```) fence.
      3. Fall back to brace-balanced extraction at arbitrary depth and
         take the LAST plausible object — LLMs sometimes echo the schema
         template before answering, and the actual answer is the trailing
         block. Returns None when no valid JSON object can be extracted.
    """
    raw = raw.strip()
    if not raw:
        return None

    # 1. straight parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. ```json … ``` fences (or generic ``` … ```). Non-greedy on
    # purpose so we extract the *first* fenced block, which is the
    # convention when models emit a fenced answer.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. brace-balanced sweep — supports arbitrary depth (the rule
    # schema is nested 3-4 levels deep, which a depth-bounded regex
    # would silently truncate).
    for block in reversed(_extract_top_level_json_objects(raw)):
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


# ============================================================================
# Service
# ============================================================================


class RuleParserResult:
    """Plain dataclass-style container for the parsed result.

    Kept simple (no Pydantic) because the route layer maps it into the
    public `ParseNLRuleResponse` model.
    """

    __slots__ = ("draft", "errors", "clarifications", "raw_llm_output")

    def __init__(
        self,
        draft: DedupRule | None,
        errors: list[str],
        clarifications: list[str],
        raw_llm_output: str,
    ) -> None:
        self.draft = draft
        self.errors = errors
        self.clarifications = clarifications
        self.raw_llm_output = raw_llm_output


class RuleParserService:
    """Converts NL strings into structured DedupRule via LLM."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def parse_nl_to_rule(
        self,
        nl: str,
        available_integrations: list[str] | None = None,
        available_slack_channels: list[str] | None = None,
        available_email_templates: list[str] | None = None,
    ) -> RuleParserResult:
        prompt = build_prompt(
            nl_input=nl,
            available_integrations=available_integrations or [],
            available_slack_channels=available_slack_channels or [],
            available_email_templates=available_email_templates or [],
        )

        # Low temperature: we want schema adherence, not creativity.
        # Wrap the LLM call so a provider outage / timeout produces a
        # structured parser error rather than bubbling a 500 up to the
        # admin UI. The exception details are logged but not surfaced —
        # they can include backend URLs / auth headers in some clients.
        try:
            raw = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=800,
            )
        except Exception:
            logger.exception("LLM generate failed during rule parse")
            return RuleParserResult(
                draft=None,
                errors=["Failed to generate a rule draft. Please retry."],
                clarifications=[],
                raw_llm_output="",
            )

        envelope = _extract_json_object(raw)
        if envelope is None:
            return RuleParserResult(
                draft=None,
                errors=["LLM did not return a parseable JSON envelope."],
                clarifications=[],
                raw_llm_output=raw,
            )

        errors_from_llm = _to_str_list(envelope.get("errors"))
        clarifications_from_llm = _to_str_list(envelope.get("clarifications"))
        draft_data = envelope.get("draft")

        # `null` draft is a legitimate signal — surface the LLM's reasoning
        if draft_data is None:
            return RuleParserResult(
                draft=None,
                errors=errors_from_llm
                or ["LLM declined to produce a rule but gave no explanation."],
                clarifications=clarifications_from_llm,
                raw_llm_output=raw,
            )

        # Validate the structured draft against the Pydantic schema
        try:
            draft = DedupRule.model_validate(draft_data)
        except ValidationError as ve:
            # Validation errors are structured — log those. Do NOT log the
            # raw LLM response: it echoes whatever the admin typed plus the
            # LLM's interpretation, which is unnecessary for diagnosing a
            # schema mismatch and risks pulling sensitive text into the
            # server log stream. The raw output is still returned in the
            # response for client-side debugging.
            logger.info(
                "LLM produced a draft that failed schema validation",
                extra={"errors": ve.errors()},
            )
            return RuleParserResult(
                draft=None,
                errors=_friendly_validation_errors(ve),
                clarifications=clarifications_from_llm,
                raw_llm_output=raw,
            )

        return RuleParserResult(
            draft=draft,
            errors=errors_from_llm,
            clarifications=clarifications_from_llm,
            raw_llm_output=raw,
        )


def _to_str_list(value: object, limit: int = 10) -> list[str]:
    """Normalize a JSON-envelope field into a bounded list[str].

    The LLM is *asked* to return arrays for `errors` and `clarifications`,
    but it sometimes returns a string ("the rule is too vague") or null.
    Without normalization, the previous code did `[str(x) for x in value]`
    which iterates a string character-by-character — producing nonsense
    output to the client. This helper accepts a list, a scalar, or null,
    and always returns a clean list bounded by `limit`.
    """
    if isinstance(value, list):
        return [str(v) for v in value][:limit]
    if value is None:
        return []
    return [str(value)][:limit]


def _friendly_validation_errors(ve: ValidationError) -> list[str]:
    """Render Pydantic validation errors into something a human can act on."""
    msgs: list[str] = []
    for err in ve.errors()[:8]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msgs.append(f"{loc}: {err.get('msg', 'invalid value')}")
    return msgs
