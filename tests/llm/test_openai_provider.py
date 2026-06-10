import types

import pytest

from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.llm.openai_provider import OpenAIProvider


def _fake_response(text="Hello!", prompt_tokens=11, completion_tokens=22):
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return types.SimpleNamespace(choices=[choice], usage=usage)


def _provider(model="gpt-4"):
    """Build an OpenAIProvider with a captured, network-free client."""
    settings = Settings(llm_provider="openai", openai_api_key="sk-test", openai_model=model)
    provider = OpenAIProvider(settings)
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    provider.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )
    return provider, captured


class TestOpenAIProviderUnit:
    def test_requires_api_key(self):
        settings = Settings(llm_provider="openai", openai_api_key=None)
        with pytest.raises(ValueError) as exc_info:
            OpenAIProvider(settings)
        assert "openai_api_key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_simple(self):
        provider, captured = _provider()
        text = await provider.generate(prompt="Say hello", temperature=0.3, max_tokens=10)
        assert text == "Hello!"
        assert captured["model"] == "gpt-4"
        assert captured["temperature"] == 0.3
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
    async def test_error_handling(self):
        import openai

        settings = Settings(llm_provider="openai", openai_api_key="sk-test")
        provider = OpenAIProvider(settings)

        async def boom(**kwargs):
            raise openai.OpenAIError("boom")

        provider.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=boom)
            )
        )
        with pytest.raises(RuntimeError) as exc_info:
            await provider.generate("test")
        assert "OpenAI API error" in str(exc_info.value)
