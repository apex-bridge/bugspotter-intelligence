from bugspotter_intelligence.observability.pricing import price_micros


class TestPriceMicros:
    def test_known_anthropic_sonnet(self):
        # 1000 input × $3/Mtok + 500 output × $15/Mtok = $0.003 + $0.0075 = $0.0105
        assert price_micros("anthropic", "claude-sonnet-4-6", 1000, 500) == 10_500

    def test_known_openai_gpt_4o_mini(self):
        # 1000 × $0.15/Mtok + 500 × $0.60/Mtok = $0.00015 + $0.00030 = $0.00045
        assert price_micros("openai", "gpt-4o-mini", 1000, 500) == 450

    def test_model_with_dated_suffix_matches_prefix(self):
        # Anthropic models often have dated suffixes; prefix-match catches them.
        assert price_micros("anthropic", "claude-sonnet-4-6-20260301", 1000, 0) == 3_000

    def test_unknown_model_returns_none(self):
        assert price_micros("ollama", "gemma3:12b", 1000, 500) is None

    def test_provider_mismatch_returns_none(self):
        # gpt-4o is a real model, but provider must match the (provider, prefix) tuple.
        assert price_micros("anthropic", "gpt-4o", 1000, 500) is None

    def test_both_token_counts_none_returns_none(self):
        assert price_micros("openai", "gpt-4o", None, None) is None

    def test_only_input_tokens(self):
        assert price_micros("openai", "gpt-4o", 1_000_000, None) == 2_500_000

    def test_only_output_tokens(self):
        assert price_micros("openai", "gpt-4o", None, 1_000_000) == 10_000_000

    def test_zero_tokens_treated_as_falsy(self):
        # 0 tokens contributes 0; result reflects only the non-zero side.
        assert price_micros("openai", "gpt-4o", 0, 500) == 5_000
