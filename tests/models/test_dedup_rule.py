"""Unit tests for the DedupRule schema.

Most of the schema's behaviour is validated indirectly through the parser
tests, but a few validation rules deserve focused coverage:
  - ActionSlack requires exactly one of `channel` / `user`
  - The discriminated unions reject unknown `type` values
"""

import pytest
from pydantic import ValidationError

from bugspotter_intelligence.models.dedup_rule import (
    ActionSlack,
    DedupRule,
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
