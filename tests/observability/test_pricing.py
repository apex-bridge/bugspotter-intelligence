from bugspotter_intelligence.observability.pricing import price_micros


class TestPriceMicros:
    def test_known_claude_sonnet(self):
        # 1000 input x $3/Mtok + 500 output x $15/Mtok = $0.003 + $0.0075 = $0.0105
        assert price_micros("claude", "claude-sonnet-4-6", 1000, 500) == 10_500

    def test_known_openai_gpt_4o_mini(self):
        # 1000 x $0.15/Mtok + 500 x $0.60/Mtok = $0.00015 + $0.00030 = $0.00045
        assert price_micros("openai", "gpt-4o-mini", 1000, 500) == 450

    def test_model_with_dated_suffix_matches_prefix(self):
        # Claude models often have dated suffixes; prefix-match catches them.
        assert price_micros("claude", "claude-sonnet-4-6-20260301", 1000, 0) == 3_000

    def test_unknown_model_returns_none(self):
        assert price_micros("ollama", "gemma3:12b", 1000, 500) is None

    def test_provider_mismatch_returns_none(self):
        # gpt-4o is a real model, but provider must match the (provider, prefix) tuple.
        assert price_micros("claude", "gpt-4o", 1000, 500) is None

    def test_both_token_counts_none_returns_none(self):
        assert price_micros("openai", "gpt-4o", None, None) is None

    def test_only_input_tokens(self):
        assert price_micros("openai", "gpt-4o", 1_000_000, None) == 2_500_000

    def test_only_output_tokens(self):
        assert price_micros("openai", "gpt-4o", None, 1_000_000) == 10_000_000

    def test_zero_tokens_treated_as_falsy(self):
        # 0 tokens contributes 0; result reflects only the non-zero side.
        assert price_micros("openai", "gpt-4o", 0, 500) == 5_000


class TestPricingMatchesResolvedProviderName:
    """The pricing-map keys must match the provider name the recorder emits.

    Regression: recorder._resolve_provider_name() derives the provider from the
    class name (ClaudeProvider -> "claude"), but the pricing map keyed Anthropic
    under "anthropic" — so every real Claude call recorded cost_micros=None even
    though tokens were captured. price_micros() in isolation looked fine because
    the old tests passed "anthropic" directly, a string the recorder never emits.
    """

    @staticmethod
    def _resolved_name(provider_cls) -> str:
        # Call the real resolver (no mirror → no drift). object.__new__ skips
        # __init__ so we don't need settings or API keys.
        from bugspotter_intelligence.observability.recorder import _resolve_provider_name

        return _resolve_provider_name(object.__new__(provider_cls))

    def test_claude_provider_name_prices(self):
        from bugspotter_intelligence.llm.claude import ClaudeProvider

        name = self._resolved_name(ClaudeProvider)
        assert name == "claude"
        # Must price under the name the recorder actually passes — not "anthropic".
        assert price_micros(name, "claude-sonnet-4-6", 1000, 500) == 10_500

    def test_openai_provider_name_prices(self):
        from bugspotter_intelligence.llm.openai_provider import OpenAIProvider

        name = self._resolved_name(OpenAIProvider)
        assert name == "openai"
        assert price_micros(name, "gpt-4o-mini", 1000, 500) == 450
