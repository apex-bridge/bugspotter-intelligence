import re
from typing import Optional

import anthropic
from anthropic import AsyncAnthropic

from .base import LLMProvider, Usage
from .factory import register_provider

# Opus 4.7 and every later Opus revision reject the sampling parameters
# (temperature/top_p/top_k) with a 400 — those models use adaptive thinking
# instead. Match the opus-4-<rev> family with rev >= 7. The `(?!\d)` guards the
# dated Opus 4.0 id (claude-opus-4-20250514), whose 8-digit date must not be
# read as a revision; every other Claude model (Sonnet 4.6, Opus 4.0/4.1/4.5/4.6)
# still accepts temperature.
_OPUS_REV = re.compile(r"opus-4-(\d{1,2})(?!\d)")


def _rejects_sampling_params(model: str) -> bool:
    match = _OPUS_REV.search(model)
    return bool(match) and int(match.group(1)) >= 7


@register_provider("claude")
class ClaudeProvider(LLMProvider):
    """LLM provider using Anthropic's Claude models via the Anthropic SDK."""

    def __init__(self, settings):
        super().__init__(settings)
        api_key = settings.anthropic_api_key
        if not api_key:
            raise ValueError(
                "anthropic_api_key (ANTHROPIC_API_KEY) is required when "
                "LLM_PROVIDER=claude"
            )
        self.client = AsyncAnthropic(api_key=api_key, timeout=settings.claude_timeout)

    async def generate(
            self,
            prompt: str,
            context: Optional[list[str]] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> str:
        """Generate a response from Claude"""
        text, _ = await self.generate_with_usage(prompt, context, temperature, max_tokens)
        return text

    async def generate_with_usage(
            self,
            prompt: str,
            context: Optional[list[str]] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000,
    ) -> tuple[str, Usage]:
        full_prompt = self._build_context_prompt(prompt, context)
        model = self.settings.claude_model

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": full_prompt}],
        }
        if not _rejects_sampling_params(model):
            kwargs["temperature"] = temperature

        try:
            response = await self.client.messages.create(**kwargs)
        except anthropic.APIError as e:
            raise RuntimeError(f"Claude API error: {e}") from e

        # response.content is a list of content blocks; concatenate the text ones.
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        usage = Usage(
            input=response.usage.input_tokens,
            output=response.usage.output_tokens,
        )
        return text, usage
