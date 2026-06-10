from typing import Optional

import anthropic
from anthropic import AsyncAnthropic

from .base import LLMProvider, Usage
from .factory import register_provider

# Opus 4.7 and later removed the sampling parameters (temperature/top_p/top_k)
# and return a 400 if they are sent — those models use adaptive thinking
# instead. Skip `temperature` for them; every other Claude model (Sonnet 4.6,
# Sonnet 4.0, etc.) still accepts it.
_NO_SAMPLING_PARAMS = ("opus-4-7", "opus-4-8")


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
        if not any(tag in model for tag in _NO_SAMPLING_PARAMS):
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
