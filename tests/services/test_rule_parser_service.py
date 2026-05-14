"""Unit tests for RuleParserService.

We exercise:
  - happy path: LLM returns clean JSON → parsed into DedupRule
  - LLM wraps JSON in ```json``` fence → still parsed
  - LLM emits null draft + clarifications → surfaced as-is
  - LLM emits schema-invalid draft → null draft + friendly errors
  - LLM emits garbage → null draft + "no JSON envelope" error
  - tenant context (integrations / channels / templates) is injected into prompt

LLM is fully mocked — no Ollama, no network.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bugspotter_intelligence.llm import LLMProvider
from bugspotter_intelligence.models.dedup_rule import DedupRule
from bugspotter_intelligence.services.rule_parser_service import (
    RuleParserService,
    _extract_json_object,
    build_prompt,
)


def _make_llm(response: str) -> LLMProvider:
    provider = MagicMock(spec=LLMProvider)
    provider.generate = AsyncMock(return_value=response)
    return provider


# ============================================================================
# build_prompt — pure string transform, no LLM
# ============================================================================


class TestBuildPrompt:
    def test_includes_user_input(self):
        prompt = build_prompt(
            nl_input="ping me when a closed bug regresses",
            available_integrations=[],
            available_slack_channels=[],
            available_email_templates=[],
        )
        assert "ping me when a closed bug regresses" in prompt

    def test_includes_tenant_context(self):
        prompt = build_prompt(
            nl_input="dummy",
            available_integrations=["jira", "linear"],
            available_slack_channels=["#regressions", "#vip"],
            available_email_templates=["dedup_ack", "weekly_digest"],
        )
        assert "jira, linear" in prompt
        assert "#regressions, #vip" in prompt
        assert "dedup_ack, weekly_digest" in prompt

    def test_signals_when_lists_are_empty(self):
        prompt = build_prompt(
            nl_input="dummy",
            available_integrations=[],
            available_slack_channels=[],
            available_email_templates=[],
        )
        # The "(none)" signal tells the LLM not to invent targets
        assert "(none)" in prompt


# ============================================================================
# _extract_json_object — robust JSON extraction
# ============================================================================


class TestExtractJsonObject:
    def test_direct_json(self):
        assert _extract_json_object('{"a": 1}') == {"a": 1}

    def test_json_in_markdown_fence(self):
        wrapped = 'Here you go:\n```json\n{"a": 1, "b": [2]}\n```\n'
        assert _extract_json_object(wrapped) == {"a": 1, "b": [2]}

    def test_json_in_generic_fence(self):
        wrapped = "```\n{\"a\": 1}\n```"
        assert _extract_json_object(wrapped) == {"a": 1}

    def test_picks_last_block_when_schema_echoed(self):
        # LLM sometimes prints the schema template first, then the answer.
        raw = '{"template": "..."} and then {"a": 1, "b": 2}'
        assert _extract_json_object(raw) == {"a": 1, "b": 2}

    def test_returns_none_on_garbage(self):
        assert _extract_json_object("not json at all") is None

    def test_returns_none_on_empty(self):
        assert _extract_json_object("") is None
        assert _extract_json_object("   \n\t  ") is None

    def test_handles_deeply_nested_objects(self):
        # The real DedupRule envelope is 4+ levels deep
        # (draft → when → ... or draft → if → [ConditionSpec] → value-list).
        # The previous regex-based extractor only handled 2 levels.
        deep = '{"draft": {"when": {"type": "schedule", "cron": "* * * * *"}, "then": [{"type": "notify.email", "to": "x@y.z", "template": "t"}]}, "errors": [], "clarifications": []}'
        parsed = _extract_json_object(deep)
        assert parsed is not None
        assert parsed["draft"]["when"]["type"] == "schedule"

    def test_handles_braces_inside_strings(self):
        # An LLM may emit a description containing literal `{` / `}`.
        # The extractor's brace counter must ignore them inside strings.
        body = '{"draft": null, "errors": ["use `{name}` to interpolate"], "clarifications": []}'
        parsed = _extract_json_object(body)
        assert parsed is not None
        assert parsed["errors"] == ["use `{name}` to interpolate"]

    def test_picks_last_object_when_schema_echoed(self):
        # The LLM sometimes prints the schema before the answer. We want
        # the answer (the last top-level object).
        echoed = (
            'Here is the schema: {"draft": {"name": "..."}, "errors": []}\n'
            'And the answer: {"draft": {"name": "Real"}, "errors": [], "clarifications": []}'
        )
        parsed = _extract_json_object(echoed)
        assert parsed is not None
        assert parsed["draft"]["name"] == "Real"


# ============================================================================
# RuleParserService.parse_nl_to_rule
# ============================================================================


def _valid_rule_envelope() -> str:
    return """{
      "draft": {
        "name": "Auto-reopen on regression",
        "when": { "type": "outbox_about_to_skip" },
        "if": [
          { "field": "canonical.status", "op": "in", "value": ["closed", "wont_fix"] },
          { "field": "hits_in_window", "op": "gte", "value": 3, "window": "24h" }
        ],
        "then": [
          { "type": "ticket.transition", "target": "canonical", "to": "in_progress" },
          { "type": "ticket.add_comment", "target": "canonical", "body": "regression" }
        ],
        "enabled": true
      },
      "errors": [],
      "clarifications": []
    }"""


class TestRuleParserService:
    @pytest.mark.asyncio
    async def test_happy_path_produces_valid_rule(self):
        llm = _make_llm(_valid_rule_envelope())
        service = RuleParserService(llm)

        result = await service.parse_nl_to_rule(
            nl="when a closed bug gets 3 hits in 24h, reopen and comment"
        )

        assert result.draft is not None
        assert isinstance(result.draft, DedupRule)
        assert result.draft.name == "Auto-reopen on regression"
        assert result.draft.when.type == "outbox_about_to_skip"
        assert len(result.draft.if_) == 2
        assert len(result.draft.then) == 2
        assert result.errors == []
        assert result.clarifications == []

    @pytest.mark.asyncio
    async def test_json_in_markdown_fence_is_still_parsed(self):
        wrapped = "Sure thing!\n```json\n" + _valid_rule_envelope() + "\n```"
        llm = _make_llm(wrapped)
        service = RuleParserService(llm)

        result = await service.parse_nl_to_rule(nl="dummy")

        assert result.draft is not None
        assert result.draft.name == "Auto-reopen on regression"

    @pytest.mark.asyncio
    async def test_null_draft_with_clarifications_is_surfaced(self):
        envelope = """{
          "draft": null,
          "errors": ["Input is too abstract."],
          "clarifications": [
            "Which trigger? (new duplicate, cluster growing, ...)",
            "What action?"
          ]
        }"""
        llm = _make_llm(envelope)
        service = RuleParserService(llm)

        result = await service.parse_nl_to_rule(nl="make things faster")

        assert result.draft is None
        assert "abstract" in result.errors[0].lower()
        assert len(result.clarifications) == 2

    @pytest.mark.asyncio
    async def test_schema_invalid_draft_yields_friendly_errors(self):
        # `when.type` = "unknown_trigger" is not in the discriminator union.
        envelope = """{
          "draft": {
            "name": "Bad rule",
            "when": { "type": "unknown_trigger" },
            "then": [
              { "type": "notify.email", "to": "reporter", "template": "dedup_ack" }
            ]
          },
          "errors": [],
          "clarifications": []
        }"""
        llm = _make_llm(envelope)
        service = RuleParserService(llm)

        result = await service.parse_nl_to_rule(nl="dummy")

        assert result.draft is None
        assert result.errors
        assert any("when" in e for e in result.errors)
        # raw output preserved for debugging
        assert result.raw_llm_output is not None
        assert "unknown_trigger" in result.raw_llm_output

    @pytest.mark.asyncio
    async def test_invalid_action_field_is_rejected(self):
        # `to: "in_progress"` is not a valid status for transition? Actually it
        # is. Use an actually-invalid one: `to: "deleted"`.
        envelope = """{
          "draft": {
            "name": "Bad transition",
            "when": { "type": "duplicate_detected" },
            "then": [
              { "type": "ticket.transition", "target": "canonical", "to": "deleted" }
            ]
          },
          "errors": [],
          "clarifications": []
        }"""
        llm = _make_llm(envelope)
        service = RuleParserService(llm)

        result = await service.parse_nl_to_rule(nl="dummy")

        assert result.draft is None
        assert result.errors

    @pytest.mark.asyncio
    async def test_garbage_output_yields_envelope_error(self):
        llm = _make_llm("I'm sorry Dave, I'm afraid I can't do that.")
        service = RuleParserService(llm)

        result = await service.parse_nl_to_rule(nl="dummy")

        assert result.draft is None
        assert len(result.errors) == 1
        assert "JSON" in result.errors[0]

    @pytest.mark.asyncio
    async def test_then_must_have_at_least_one_action(self):
        # Schema requires `then` to be non-empty (min_length=1).
        envelope = """{
          "draft": {
            "name": "Empty actions",
            "when": { "type": "duplicate_detected" },
            "then": []
          },
          "errors": [],
          "clarifications": []
        }"""
        llm = _make_llm(envelope)
        service = RuleParserService(llm)

        result = await service.parse_nl_to_rule(nl="dummy")

        assert result.draft is None
        assert any("then" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_tenant_context_is_passed_to_prompt(self):
        # Sanity: when the user supplies channels, those land in the LLM prompt.
        captured: list[str] = []

        async def capture_generate(prompt: str, **kwargs):
            captured.append(prompt)
            return _valid_rule_envelope()

        llm = MagicMock(spec=LLMProvider)
        llm.generate = capture_generate
        service = RuleParserService(llm)

        await service.parse_nl_to_rule(
            nl="dummy",
            available_integrations=["jira"],
            available_slack_channels=["#regressions"],
            available_email_templates=["dedup_ack"],
        )

        assert captured, "LLM.generate was not invoked"
        prompt = captured[0]
        assert "jira" in prompt
        assert "#regressions" in prompt
        assert "dedup_ack" in prompt
