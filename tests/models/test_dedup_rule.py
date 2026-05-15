"""Unit tests for the DedupRule schema.

Most of the schema's behaviour is validated indirectly through the parser
tests, but a few validation rules deserve focused coverage:
  - ActionSlack requires exactly one of `channel` / `user`
  - The discriminated unions reject unknown `type` values
"""

import pytest
from pydantic import ValidationError

from bugspotter_intelligence.models.dedup_rule import (
    ActionEmail,
    ActionSlack,
    ActionWebhook,
    ConditionSpec,
    DedupRule,
    RateLimit,
    TriggerClusterGrowing,
    TriggerSchedule,
)


class TestActionSlack:
    def test_accepts_channel_only(self):
        a = ActionSlack(channel="#regressions", message="hi")
        assert a.channel == "#regressions"
        assert a.user is None

    def test_accepts_user_only(self):
        a = ActionSlack(user="closer", message="hi")
        assert a.user == "closer"
        assert a.channel is None

    def test_rejects_both_channel_and_user(self):
        with pytest.raises(ValidationError, match="exactly one"):
            ActionSlack(channel="#x", user="alice", message="hi")

    def test_rejects_neither(self):
        with pytest.raises(ValidationError, match="exactly one"):
            ActionSlack(message="hi")

    def test_rejects_empty_strings(self):
        # Whitespace-only fields don't count as "set" — otherwise the LLM
        # could trick the validator by emitting `"channel": " "`.
        with pytest.raises(ValidationError, match="exactly one"):
            ActionSlack(channel="   ", user="", message="hi")


class TestDedupRuleDiscriminator:
    def test_rejects_unknown_trigger_type(self):
        with pytest.raises(ValidationError):
            DedupRule.model_validate(
                {
                    "name": "x",
                    "when": {"type": "made_up"},
                    "then": [
                        {"type": "notify.email", "to": "reporter", "template": "ack"}
                    ],
                }
            )

    def test_rejects_unknown_action_type(self):
        with pytest.raises(ValidationError):
            DedupRule.model_validate(
                {
                    "name": "x",
                    "when": {"type": "duplicate_detected"},
                    "then": [{"type": "ticket.delete"}],
                }
            )

    def test_then_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            DedupRule.model_validate(
                {
                    "name": "x",
                    "when": {"type": "duplicate_detected"},
                    "then": [],
                }
            )

    def test_accepts_both_if_and_if_aliases(self):
        # Wire shape uses "if"; python attribute is "if_".
        # `populate_by_name=True` means either form should construct the
        # same object — test BOTH to lock that in.
        from_wire = DedupRule.model_validate(
            {
                "name": "x",
                "when": {"type": "duplicate_detected"},
                "if": [{"field": "severity", "op": "eq", "value": "high"}],
                "then": [
                    {"type": "notify.email", "to": "reporter", "template": "ack"}
                ],
            }
        )
        assert len(from_wire.if_) == 1
        assert from_wire.if_[0].field == "severity"

        from_python = DedupRule.model_validate(
            {
                "name": "x",
                "when": {"type": "duplicate_detected"},
                "if_": [{"field": "severity", "op": "eq", "value": "high"}],
                "then": [
                    {"type": "notify.email", "to": "reporter", "template": "ack"}
                ],
            }
        )
        assert len(from_python.if_) == 1
        assert from_python.if_[0].field == "severity"


class TestWindowPattern:
    """The `window` fields (trigger / condition / rate_limit) accept only
    `<digits><unit>` strings — the executor parses these into seconds, and
    accepting "an hour" or "1 hr" would just be a delayed failure.
    """

    @pytest.mark.parametrize("good", ["1s", "30s", "1h", "24h", "7d", "1w", "100h"])
    def test_trigger_window_accepts_well_formed(self, good: str):
        TriggerClusterGrowing(threshold=5, window=good)

    @pytest.mark.parametrize("bad", ["", "1", "1 h", "1hr", "one hour", "1m30s", "h1"])
    def test_trigger_window_rejects_malformed(self, bad: str):
        with pytest.raises(ValidationError):
            TriggerClusterGrowing(threshold=5, window=bad)

    @pytest.mark.parametrize("bad", ["0s", "0h", "0d", "00h"])
    def test_trigger_window_rejects_zero_duration(self, bad: str):
        # Zero-duration windows would make conditions meaningless and
        # rate-limits permissive — reject at parse time.
        with pytest.raises(ValidationError):
            TriggerClusterGrowing(threshold=5, window=bad)

    def test_rate_limit_window_pattern(self):
        RateLimit(count=1, window="1h")  # ok
        with pytest.raises(ValidationError):
            RateLimit(count=1, window="every hour")


class TestTriggerScheduleCron:
    """Cron pattern catches obvious prose / typos. Semantic validity is
    delegated to the executor — the schema's job is to stop "every monday"
    from sneaking through the parse loop."""

    @pytest.mark.parametrize(
        "good", ["0 9 * * 1", "*/5 * * * *", "0 0 1 1 *", "30 9 * * 1-5"]
    )
    def test_accepts_well_formed_cron(self, good: str):
        TriggerSchedule(cron=good)

    @pytest.mark.parametrize(
        "bad", ["every monday", "0 9 * *", "0 9 * * 1 0", "", "monday at 9"]
    )
    def test_rejects_malformed_cron(self, bad: str):
        with pytest.raises(ValidationError):
            TriggerSchedule(cron=bad)


class TestConditionSpec:
    def test_accepts_known_field(self):
        ConditionSpec(field="canonical.status", op="in", value=["closed"])

    def test_rejects_unknown_field(self):
        # LLMs hallucinate field names that look plausible but aren't
        # supported by the executor — schema-level closure catches them.
        with pytest.raises(ValidationError):
            ConditionSpec(field="canonical.priority", op="eq", value="high")

    def test_in_op_rejects_non_coercible_value(self):
        # Scalars are coerced (covered by test_in_op_coerces_scalar_to_singleton_list);
        # only non-scalar, non-list values like dicts should be rejected.
        with pytest.raises(ValidationError, match="requires a list value"):
            ConditionSpec(field="canonical.status", op="in", value={"bad": True})

    def test_not_in_op_rejects_non_coercible_value(self):
        with pytest.raises(ValidationError, match="requires a list value"):
            ConditionSpec(field="severity", op="not_in", value={"bad": True})

    def test_gte_op_requires_number(self):
        with pytest.raises(ValidationError, match="requires a numeric value"):
            ConditionSpec(
                field="hits_in_window", op="gte", value="three", window="24h"
            )

    def test_eq_op_accepts_scalar(self):
        # No mismatch on `eq` — it's the catch-all op for single values.
        ConditionSpec(field="reporter.customer.tier", op="eq", value="enterprise")

    def test_in_op_coerces_scalar_to_singleton_list(self):
        # LLMs frequently emit `"value": "closed"` for `op: in`. Coerce
        # silently — the alternative is a re-parse loop with no useful
        # signal. The condition is logically equivalent.
        c = ConditionSpec(field="canonical.status", op="in", value="closed")
        assert c.value == ["closed"]

        c2 = ConditionSpec(field="canonical.closed_days_ago", op="in", value=7)
        assert c2.value == [7]

    def test_gte_coerces_string_number(self):
        # Same coercion logic for `"3"` → 3.0 on gte/lte. Real LLM output.
        c = ConditionSpec(field="hits_in_window", op="gte", value="3", window="24h")
        assert c.value == 3.0

    def test_gte_rejects_non_numeric_string(self):
        with pytest.raises(ValidationError, match="requires a numeric value"):
            ConditionSpec(
                field="hits_in_window", op="gte", value="lots", window="24h"
            )

    def test_gte_rejects_bool(self):
        # `True > 5` evaluates in Python — reject explicitly so a bool
        # smuggled in via JSON doesn't silently pass.
        with pytest.raises(ValidationError, match="requires a numeric value"):
            ConditionSpec(field="hits_in_window", op="gte", value=True, window="24h")

    def test_hits_in_window_requires_window_parameter(self):
        # The prompt says so; the executor enforces it. Surface at parse
        # time so the LLM sees the feedback in the same loop.
        with pytest.raises(ValidationError, match="requires a 'window' parameter"):
            ConditionSpec(field="hits_in_window", op="gte", value=3)


class TestActionEmailTo:
    """Recipient validation accepts the three special tokens or an
    email-shaped string. Strict RFC-5322 is the executor's job."""

    @pytest.mark.parametrize(
        "good",
        [
            "reporter",
            "closer",
            "all_reporters",
            "team@example.com",
            "alice+ops@sub.example.co.uk",
        ],
    )
    def test_accepts_valid_targets(self, good: str):
        ActionEmail(to=good, template="dedup_ack")

    @pytest.mark.parametrize(
        "bad",
        [
            "Reporter",  # case-sensitive special tokens — match the executor's exact set
            "everyone",
            "not-an-email",
            "@example.com",
            "alice@",
            "alice@nodot",
        ],
    )
    def test_rejects_invalid_targets(self, bad: str):
        with pytest.raises(ValidationError):
            ActionEmail(to=bad, template="dedup_ack")


class TestActionWebhookUrl:
    """Pydantic's HttpUrl rejects malformed URLs at schema-time so the LLM
    can't smuggle in a notify.webhook with a bad target that only fails
    at executor time.
    """

    def test_accepts_valid_https_url(self):
        ActionWebhook(url="https://example.com/hook")

    def test_accepts_valid_http_url(self):
        ActionWebhook(url="http://localhost:8080/hook")

    def test_rejects_non_url_string(self):
        with pytest.raises(ValidationError):
            ActionWebhook(url="not-a-url")

    def test_rejects_javascript_scheme(self):
        # HttpUrl restricts scheme to http/https — the LLM shouldn't be
        # able to emit a javascript: target even by mistake.
        with pytest.raises(ValidationError):
            ActionWebhook(url="javascript:alert(1)")
