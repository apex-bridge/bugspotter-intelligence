"""Tests for OllamaProvider.generate_with_usage — token + duration extraction + error paths."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.llm.ollama import OllamaProvider


@pytest.fixture
def provider():
    return OllamaProvider(Settings())


@pytest.mark.asyncio
async def test_returns_token_counts_from_ollama_response(provider):
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "answer",
            "prompt_eval_count": 12,
            "eval_count": 34,
            "eval_duration": 567_000_000,
            "prompt_eval_duration": 89_000_000,
            "total_duration": 700_000_000,
            "load_duration": 1_000_000,
        }
        mock_post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = mock_post

        text, usage = await provider.generate_with_usage(prompt="hi", max_tokens=10)

    assert text == "answer"
    assert usage.input == 12
    assert usage.output == 34
    assert usage.extra["eval_duration"] == 567_000_000
    assert usage.extra["prompt_eval_duration"] == 89_000_000
    assert usage.extra["total_duration"] == 700_000_000
    assert usage.extra["load_duration"] == 1_000_000


@pytest.mark.asyncio
async def test_handles_missing_usage_fields(provider):
    """Some Ollama versions / models omit eval_* fields — must not crash."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "answer"}
        mock_post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = mock_post

        text, usage = await provider.generate_with_usage(prompt="hi")

    assert text == "answer"
    assert usage.input is None
    assert usage.output is None
    assert usage.extra == {}


@pytest.mark.asyncio
async def test_generate_still_returns_just_text(provider):
    """The legacy generate() contract is preserved by delegating to generate_with_usage()."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "answer", "eval_count": 5}
        mock_post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = mock_post

        text = await provider.generate(prompt="hi")

    assert text == "answer"


@pytest.mark.asyncio
async def test_http_error_wrapped_with_cause(provider):
    """5xx from Ollama → RuntimeError with original HTTPStatusError as __cause__."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "internal server error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=mock_response,
        )
        mock_post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = mock_post

        with pytest.raises(RuntimeError, match="500") as ei:
            await provider.generate_with_usage(prompt="hi")

    assert isinstance(ei.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.asyncio
async def test_malformed_json_body_wrapped_with_cause(provider):
    """JSONDecodeError → RuntimeError that mentions the raw body and preserves cause."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "not actually json"
        mock_response.json.side_effect = json.JSONDecodeError("expecting value", "doc", 0)
        mock_post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = mock_post

        with pytest.raises(RuntimeError, match="Unexpected Ollama response format") as ei:
            await provider.generate_with_usage(prompt="hi")

    assert isinstance(ei.value.__cause__, ValueError)  # JSONDecodeError is a ValueError


@pytest.mark.asyncio
async def test_missing_response_key_wrapped_with_cause(provider):
    """KeyError on result['response'] → RuntimeError with KeyError as cause."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"prompt_eval_count": 1}'
        mock_response.json.return_value = {"prompt_eval_count": 1}
        mock_post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = mock_post

        with pytest.raises(RuntimeError, match="Unexpected Ollama response format") as ei:
            await provider.generate_with_usage(prompt="hi")

    assert isinstance(ei.value.__cause__, KeyError)
