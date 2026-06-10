import types

import pytest

from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.llm.claude import ClaudeProvider, _rejects_sampling_params


@pytest.mark.parametrize(
    "model,rejects",
    [
        ("claude-opus-4-7", True),
        ("claude-opus-4-8", True),
        ("claude-opus-4-9", True),   # future revision, must also be gated
        ("claude-opus-4-10", True),
        ("claude-opus-4-6", False),
        ("claude-opus-4-5", False),
        ("claude-opus-4-1-20250805", False),   # Opus 4.1 accepts temperature
        ("claude-opus-4-20250514", False),     # legacy Opus 4.0 (dated id, not a revision)
        ("claude-sonnet-4-6", False),
        ("gpt-4", False),
    ],
)
def test_rejects_sampling_params(model, rejects):
    assert _rejects_sampling_params(model) is rejects


def _fake_response(text="Hello!", input_tokens=11, output_tokens=22):
    block = types.SimpleNamespace(type="text", text=text)
    usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return types.SimpleNamespace(content=[block], usage=usage)


def _provider(model="claude-sonnet-4-6"):
    """Build a ClaudeProvider with a captured, network-free client."""
    settings = Settings(llm_provider="claude", anthropic_api_key="sk-test", claude_model=model)
    provider = ClaudeProvider(settings)
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    provider.client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=fake_create)
    )
    return provider, captured


class TestClaudeProviderUnit:
    def test_requires_api_key(self):
        settings = Settings(llm_provider="claude", anthropic_api_key=None)
        with pytest.raises(ValueError) as exc_info:
            ClaudeProvider(settings)
        assert "anthropic_api_key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_simple(self):
        provider, captured = _provider()
        text = await provider.generate(prompt="Say hello", max_tokens=10)
        assert text == "Hello!"
        assert captured["model"] == "claude-sonnet-4-6"
        assert captured["max_tokens"] == 10
        assert captured["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_generate_with_usage(self):
        provider, _ = _provider()
        text, usage = await provider.generate_with_usage(prompt="Q", max_tokens=10)
        assert text == "Hello!"
        assert usage.input == 11
        assert usage.output == 22

    @pytest.mark.asyncio
    async def test_context_is_included_in_prompt(self):
        provider, captured = _provider()
        await provider.generate_with_usage(prompt="What's the issue?", context=["Bug #1: crash"])
        sent = captured["messages"][0]["content"]
        assert "Bug #1: crash" in sent

    @pytest.mark.asyncio
    async def test_sampling_param_sent_for_sonnet(self):
        provider, captured = _provider("claude-sonnet-4-6")
        await provider.generate_with_usage(prompt="Q", temperature=0.3)
        assert captured["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_sampling_param_skipped_for_opus_4_8(self):
        # Opus 4.7+ reject temperature/top_p/top_k with a 400; the provider must omit them.
        provider, captured = _provider("claude-opus-4-8")
        await provider.generate_with_usage(prompt="Q", temperature=0.3)
        assert "temperature" not in captured

    @pytest.mark.asyncio
    async def test_error_handling(self):
        import anthropic

        settings = Settings(llm_provider="claude", anthropic_api_key="sk-test")
        provider = ClaudeProvider(settings)

        async def boom(**kwargs):
            raise anthropic.APIError("boom", request=None, body=None)

        provider.client = types.SimpleNamespace(
            messages=types.SimpleNamespace(create=boom)
        )
        with pytest.raises(RuntimeError) as exc_info:
            await provider.generate("test")
        assert "Claude API error" in str(exc_info.value)
