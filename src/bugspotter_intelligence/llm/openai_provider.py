from typing import Optional

import openai
from openai import AsyncOpenAI

from .base import LLMProvider, Usage
from .factory import register_provider


@register_provider("openai")
class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI's chat completion models."""

    def __init__(self, settings):
        super().__init__(settings)
        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError(
                "openai_api_key (OPENAI_API_KEY) is required when "
                "LLM_PROVIDER=openai"
            )
        self.client = AsyncOpenAI(api_key=api_key, timeout=settings.openai_timeout)

    async def generate(
            self,
            prompt: str,
            context: Optional[list[str]] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> str:
        """Generate a response from OpenAI"""
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

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

        text = (response.choices[0].message.content or "") if response.choices else ""

        usage = Usage(
            input=response.usage.prompt_tokens if response.usage else None,
            output=response.usage.completion_tokens if response.usage else None,
        )
        return text, usage
