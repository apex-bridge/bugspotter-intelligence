"""LLM-based reranker for smart search mode"""

import asyncio
import json
import logging
import re
from typing import Optional

from bugspotter_intelligence.llm import LLMProvider

logger = logging.getLogger(__name__)

_RERANK_PROMPT_TEMPLATE = """Score each bug report's relevance to the search query on a scale of 0.0 (completely irrelevant) to 1.0 (perfect match).

Search query: "{query}"

{candidates_text}

Return ONLY a JSON array of {count} floats in the same order as the candidates above. Example: [0.9, 0.3, 0.7]"""


class LLMReranker:
    """
    Reranks search results using LLM relevance scoring.

    Sends candidate results to the LLM for batch scoring,
    then sorts by LLM-assigned relevance scores.
    Falls back to original ordering on timeout or error.
    """

    def __init__(self, llm_provider: LLMProvider, timeout_seconds: float = 10.0):
        self.llm_provider = llm_provider
        self.timeout_seconds = timeout_seconds

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        return_limit: int = 5,
    ) -> tuple[list[dict], bool]:
        """
        Rerank candidates using LLM relevance scoring.

        Args:
            query: Original search query
            candidates: List of result dicts from fast search
            return_limit: Number of results to return after reranking

        Returns:
            Tuple of (reranked results, llm_used flag).
            llm_used is False if fallback to original ordering was used.
        """
        if not candidates:
            return [], True

        prompt = self._build_prompt(query, candidates)

        try:
            raw_response = await asyncio.wait_for(
                self.llm_provider.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=200,
                ),
                timeout=self.timeout_seconds,
            )

            scores = self._parse_scores(raw_response, len(candidates))

            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for candidate, score in scored[:return_limit]:
                result = dict(candidate)
                result["similarity"] = score
                reranked.append(result)

            return reranked, True

        except asyncio.TimeoutError:
            logger.warning(
                f"LLM reranking timed out after {self.timeout_seconds}s, "
                "falling back to original ordering"
            )
            return candidates[:return_limit], False

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}, falling back to original ordering")
            return candidates[:return_limit], False

    def _build_prompt(self, query: str, candidates: list[dict]) -> str:
        """Build the scoring prompt for the LLM."""
        lines = []
        for i, c in enumerate(candidates, 1):
            title = c.get("title", "")
            desc = c.get("description") or ""
            status = c.get("status", "")
            resolution = c.get("resolution") or ""

            parts = [f"Candidate {i}: [{status}] {title}"]
            if desc:
                truncated_desc = desc[:200]
                if len(desc) > 200:
                    truncated_desc += "..."
                parts.append(f"  Description: {truncated_desc}")
            if resolution:
                truncated_res = resolution[:100]
                if len(resolution) > 100:
                    truncated_res += "..."
                parts.append(f"  Resolution: {truncated_res}")
            lines.append("\n".join(parts))

        candidates_text = "\n\n".join(lines)

        return _RERANK_PROMPT_TEMPLATE.format(
            query=query,
            candidates_text=candidates_text,
            count=len(candidates),
        )

    @staticmethod
    def _parse_scores(raw: str, expected_count: int) -> list[float]:
        """
        Parse LLM response into a list of scores.

        Extracts a JSON array from the response using regex pattern matching,
        clamps values to [0.0, 1.0]. Falls back to 0.5 for all candidates on
        parse failure.
        """
        text = raw.strip()

        # Use regex to find JSON array patterns (more robust than simple find/rfind)
        # Pattern matches: [ ... ] allowing any content (numbers, negatives, strings, etc.)
        array_pattern = r'\[[^\[\]]*\]'
        matches = re.findall(array_pattern, text)

        if not matches:
            logger.debug(f"No JSON array found in LLM response: {text[:100]}")
            return [0.5] * expected_count

        # Try parsing each match until we find a valid list
        for match in matches:
            try:
                scores = json.loads(match)
                if isinstance(scores, list) and len(scores) > 0:
                    # Found a valid array, use it
                    break
            except json.JSONDecodeError:
                continue
        else:
            # No valid JSON array found
            logger.debug(f"Failed to parse any JSON array from matches: {matches}")
            return [0.5] * expected_count

        if not isinstance(scores, list):
            return [0.5] * expected_count

        # Pad or truncate to expected count
        if len(scores) < expected_count:
            scores.extend([0.5] * (expected_count - len(scores)))
        scores = scores[:expected_count]

        # Clamp each score to [0.0, 1.0], fallback to 0.5 for non-numeric
        clamped = []
        for s in scores:
            try:
                val = float(s)
                clamped.append(max(0.0, min(1.0, val)))
            except (TypeError, ValueError):
                clamped.append(0.5)

        return clamped
