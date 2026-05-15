"""
Structured dedup-rule schema.

A `DedupRule` describes a single trigger → conditions → actions automation for
the BugSpotter dedup pipeline (e.g. "when a closed bug gets 3 hits in 24h,
reopen it and ping the closer in Slack").

This module is the canonical source for what a rule looks like across the
intelligence service and the BugSpotter backend. The NL parser produces these;
a future rule executor will consume them.

Design notes:
- Triggers / conditions / actions are tagged unions on `type` (or `field`).
  Pydantic v2 picks the right variant via the discriminator.
- Action types use a namespace prefix (`ticket.*`, `notify.*`) so platform-
  specific actions can be added later (e.g. `jira.link_to_epic`) without
  conflicting with platform-neutral ones.
- `value` on conditions is Any deliberately — different fields take different
  shapes (int for hits, list[str] for status enum). The executor validates per
  field; we don't try to encode that here.
"""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, HttpUrl, model_validator

# Duration string pattern used by `window` / `rate_limit.window` fields.
# Accepts forms like "30s", "1h", "24h", "7d", "1w" — the rule executor
# parses these into seconds. Restricting at the schema level prevents the
# LLM from emitting wild values like "an hour" or "1 hr".
_WINDOW_PATTERN = r"^\d+[smhdw]$"

# Standard 5-field cron (`min hour day month dow`). Deliberately loose —
# this catches typos / prose like "every monday" but doesn't validate
# semantics (e.g. "60 * * * *" passes regex but is bogus). The executor
# does the strict parse; this pattern's job is to stop obvious nonsense
# at the schema boundary so the LLM gets feedback during the parse loop
# rather than at execution time.
_CRON_PATTERN = r"^(\S+\s+){4}\S+$"

# Recipients accepted by ActionEmail.to: three special tokens resolved
# at execution time, or a literal email address. The email regex is
# intentionally loose ("anything @ anything . anything") — exact RFC-5322
# validation belongs in the executor where DNS / MX checks live.
_EMAIL_TARGET_PATTERN = r"^(reporter|closer|all_reporters|[^@\s]+@[^@\s]+\.[^@\s]+)$"

# ============================================================================
# Triggers — pick exactly one
# ============================================================================


class TriggerDuplicateDetected(BaseModel):
    """A bug just got `duplicate_of` set by the async dedup pipeline."""

    type: Literal["duplicate_detected"] = "duplicate_detected"


class TriggerOutboxAboutToSkip(BaseModel):
    """Outbox worker is about to skip filing because the bug is a duplicate.

    Fires *before* the external-ticket suppression actually happens, so the
    rule can decide to comment / reopen / etc.
    """

    type: Literal["outbox_about_to_skip"] = "outbox_about_to_skip"


class TriggerClusterGrowing(BaseModel):
    """A cluster (canonical + its duplicates) is growing fast."""

    type: Literal["cluster_growing"] = "cluster_growing"
    threshold: int = Field(
        ..., ge=2, description="Minimum new hits to fire (e.g. 5)"
    )
    window: str = Field(
        ...,
        pattern=_WINDOW_PATTERN,
        description='Window over which `threshold` applies, e.g. "1h", "24h"',
    )


class TriggerSchedule(BaseModel):
    """Cron-based recurring trigger (digest emails, weekly reports)."""

    type: Literal["schedule"] = "schedule"
    cron: str = Field(
        ...,
        pattern=_CRON_PATTERN,
        description='Standard 5-field cron, e.g. "0 9 * * 1" (Mondays 09:00)',
    )


TriggerSpec = Annotated[
    Union[
        TriggerDuplicateDetected,
        TriggerOutboxAboutToSkip,
        TriggerClusterGrowing,
        TriggerSchedule,
    ],
    Field(discriminator="type"),
]


# ============================================================================
# Conditions — zero or more, ANDed together
# ============================================================================


class ConditionSpec(BaseModel):
    """One AND-clause.

    `field` is a closed Literal so the LLM can't invent unsupported paths
    (e.g. `canonical.priority`) and have them quietly accepted; the
    executor expects exactly these field names and any other would be
    a silent no-op. Type compatibility (op + value shape) is partly
    enforced by `_validate_op_value` below; the rest (e.g. severity
    values must be one of {low, medium, high, critical}) is delegated
    to the executor where the canonical enums live.
    """

    field: Literal[
        "canonical.status",
        "canonical.closed_days_ago",
        "hits_in_window",
        "reporter.customer.tier",
        "severity",
    ] = Field(..., description="Dot-path of the property being tested")
    op: Literal["eq", "in", "not_in", "gte", "lte"] = Field(
        ..., description="Comparison operator"
    )
    value: Any = Field(..., description="Value to compare against")
    window: str | None = Field(
        None,
        pattern=_WINDOW_PATTERN,
        description=(
            'Time window for windowed fields (`hits_in_window`). '
            'Examples: "1h", "24h", "7d". Ignored for non-windowed fields.'
        ),
    )

    @model_validator(mode="after")
    def _validate_op_value(self) -> "ConditionSpec":
        """Reject obvious op/value mismatches at parse time, with light
        coercion for the LLM-friendly cases.

        The LLM gets these wrong most often:
          - `in` / `not_in` paired with a scalar value (should be a list)
          - `gte` / `lte` with a numeric-looking string (`"3"`)
          - `hits_in_window` without an accompanying `window`

        We coerce the scalar→list and string→number cases instead of
        rejecting outright — small models routinely emit these shapes
        for otherwise-correct rules, and forcing a re-parse loop adds
        latency for no useful signal. Anything we can't fix is still
        surfaced as a parse error with a hint about what the LLM should
        have emitted.

        Field-type validation (severity ∈ {low,medium,...}) stays in
        the executor where the canonical enums live.
        """
        # in / not_in must operate on lists. Wrap scalars instead of rejecting.
        if self.op in ("in", "not_in"):
            if not isinstance(self.value, list):
                if isinstance(self.value, (str, int, float, bool)):
                    self.value = [self.value]
                else:
                    raise ValueError(
                        f"op '{self.op}' on field '{self.field}' requires a list value, "
                        f"got {type(self.value).__name__}"
                    )

        # gte / lte must be numeric. Coerce string-numbers ("3" → 3.0).
        if self.op in ("gte", "lte"):
            if isinstance(self.value, str):
                try:
                    self.value = float(self.value)
                except ValueError:
                    pass
            if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
                # `bool` is a subclass of int in Python; reject it
                # explicitly since `True > 5` evaluates and would silently
                # pass downstream.
                raise ValueError(
                    f"op '{self.op}' on field '{self.field}' requires a numeric value, "
                    f"got {type(self.value).__name__}"
                )

        # `hits_in_window` is meaningless without a window — the prompt
        # says so and the executor enforces it. Surface here for early
        # feedback rather than rejecting at execute time.
        if self.field == "hits_in_window" and not self.window:
            raise ValueError(
                "field 'hits_in_window' requires a 'window' parameter, e.g. '24h'"
            )

        return self


# ============================================================================
# Actions — at least one
# ============================================================================


CanonicalStatus = Literal["open", "in_progress", "closed", "wont_fix"]


class ActionAddComment(BaseModel):
    """Add a comment to the canonical bug's external ticket (Jira/Linear/...)."""

    type: Literal["ticket.add_comment"] = "ticket.add_comment"
    target: Literal["canonical"] = "canonical"
    body: str = Field(
        ...,
        description=(
            "Comment body. Supports template vars like {canonical.key}, "
            "{hits_24h}, {cluster.size}."
        ),
    )


class ActionTransition(BaseModel):
    """Change the canonical ticket's status (reopen, close, etc.)."""

    type: Literal["ticket.transition"] = "ticket.transition"
    target: Literal["canonical"] = "canonical"
    to: CanonicalStatus = Field(..., description="Target canonical status")


class ActionEmail(BaseModel):
    """Send a notification email."""

    type: Literal["notify.email"] = "notify.email"
    to: str = Field(
        ...,
        pattern=_EMAIL_TARGET_PATTERN,
        description=(
            'Recipient. Special values: "reporter" (the user who submitted), '
            '"closer" (last person to close the canonical), "all_reporters" '
            '(everyone who ever reported into this cluster), or a literal '
            "email address."
        ),
    )
    template: str = Field(
        ...,
        description='Template id, e.g. "dedup_ack", "regression_alert", "weekly_digest".',
    )


class ActionSlack(BaseModel):
    """Post a Slack message."""

    type: Literal["notify.slack"] = "notify.slack"
    channel: str | None = Field(
        None,
        description='Channel name (e.g. "#regressions"). Mutually exclusive with `user`.',
    )
    user: str | None = Field(
        None,
        description=(
            'DM target. Special values: "closer" (resolves at runtime), or a Slack handle.'
        ),
    )
    message: str = Field(
        ..., description="Message body. Same template vars as ActionAddComment."
    )

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "ActionSlack":
        """Enforce: exactly one of `channel` / `user` is set.

        Without this, the LLM can emit a Slack action with both fields (or
        neither) and validation passes — only the executor would catch the
        misconfiguration at runtime, and only if it bothered to. Better to
        surface it as a parse-time schema error.
        """
        has_channel = self.channel is not None and self.channel.strip() != ""
        has_user = self.user is not None and self.user.strip() != ""
        if has_channel == has_user:
            raise ValueError(
                "ActionSlack requires exactly one of `channel` or `user` to be set"
            )
        return self


class ActionWebhook(BaseModel):
    """POST a JSON payload to a custom webhook URL."""

    type: Literal["notify.webhook"] = "notify.webhook"
    url: HttpUrl = Field(
        ...,
        description=(
            "Webhook URL. Pydantic's HttpUrl validator enforces scheme + host "
            "shape so the LLM can't emit `notify.webhook` with a malformed "
            "URL that only fails at executor time."
        ),
    )
    payload: dict[str, Any] | None = Field(
        None, description="Optional JSON body. Template vars supported in string values."
    )


ActionSpec = Annotated[
    Union[
        ActionAddComment,
        ActionTransition,
        ActionEmail,
        ActionSlack,
        ActionWebhook,
    ],
    Field(discriminator="type"),
]


# ============================================================================
# Rule
# ============================================================================


class RateLimit(BaseModel):
    """Bound how often a single rule may fire.

    Applied per (rule, canonical) pair: if a rule is `1 per hour` and the same
    canonical fires the rule twice in an hour, only the first proceeds; the
    second is dropped (or batched, executor-dependent).
    """

    count: int = Field(..., ge=1, description="Max fires per window")
    window: str = Field(..., pattern=_WINDOW_PATTERN, description='Window, e.g. "1h", "24h"')


class DedupRule(BaseModel):
    """A single trigger→conditions→actions automation."""

    name: str = Field(..., min_length=1, max_length=120, description="Human-readable label")
    when: TriggerSpec = Field(..., description="What event causes the rule to fire")
    if_: list[ConditionSpec] = Field(
        default_factory=list,
        alias="if",
        description="Optional AND'd conditions; empty list means always pass",
    )
    then: list[ActionSpec] = Field(
        ..., min_length=1, description="Actions to run when the rule fires"
    )
    rate_limit: RateLimit | None = Field(
        None, description="Optional cap on firing frequency per canonical"
    )
    enabled: bool = Field(
        default=True, description="Toggle without deleting the rule"
    )

    model_config = {
        # Allow constructing via both `if_` (python) and `if` (JSON/wire)
        "populate_by_name": True,
    }
