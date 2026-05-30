import os
import httpx
from typing import Optional
from .base import LLMProvider, Usage
from .factory import register_provider

@register_provider("ollama")
class OllamaProvider(LLMProvider):
    """LLM provider using local Ollama"""
    def __init__(self, settings):
        super().__init__(settings)
        self.API_TIMEOUT = float(os.getenv('OLLAMA_TIMEOUT', '120'))

    async def generate(
            self,
            prompt: str,
            context: Optional[list[str]] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> str:
        """Generate a response from the Ollama LLM"""
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

        payload = {
            "model": self.settings.ollama_model,
            "prompt": full_prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False
        }

        timeout = httpx.Timeout(self.API_TIMEOUT, read=self.API_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json=payload,
            )

            try:
                response.raise_for_status()
                result = response.json()
                text = result["response"]
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Ollama API error: {e.response.status_code} - {e.response.text}"
                ) from e
            except (ValueError, KeyError) as e:
                # ValueError covers JSONDecodeError (malformed body); KeyError
                # covers a response with missing "response" key.
                raise RuntimeError(
                    f"Unexpected Ollama response format: {response.text}"
                ) from e

            usage = Usage(
                input=result.get("prompt_eval_count"),
                output=result.get("eval_count"),
                extra={
                    k: result[k]
                    for k in (
                        "eval_duration",
                        "prompt_eval_duration",
                        "total_duration",
                        "load_duration",
                    )
                    if k in result
                },
            )
            return text, usage