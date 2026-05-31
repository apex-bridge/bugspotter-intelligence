from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Usage:
    """Token counts and provider extras from one LLM call; None = unknown, not zero."""
    input: Optional[int] = None
    output: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, settings):
        """Store settings in all providers"""
        self.settings = settings

    @abstractmethod
    async def generate(
            self,
            prompt: str,
            context: Optional[list[str]] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> str:
        """
        Generate a response from the LLM

        Args:
            prompt: The user's question or instruction
            context: Optional list of context strings (e.g., similar bug descriptions)
            temperature: Randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum response length

        Returns:
            Generated text response
        """
        pass

    async def generate_with_usage(
            self,
            prompt: str,
            context: Optional[list[str]] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000,
    ) -> tuple[str, Usage]:
        """Like generate(), but also returns token usage; default returns empty Usage."""
        text = await self.generate(prompt, context, temperature, max_tokens)
        return text, Usage()

    def _build_context_prompt(self, prompt: str, context: Optional[list[str]] = None) -> str:
        """
        Helper method to combine prompt with context
        This is concrete (not abstract) - all providers can use it
        """
        if not context:
            return prompt

        context_text = "\n\n".join([f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(context)])

        return f"""You are a helpful assistant analyzing bug reports.

{context_text}

User Question: {prompt}

Answer:"""