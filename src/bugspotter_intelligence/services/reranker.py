"""LLM-based reranker for smart search mode"""

import asyncio
import json
import logging

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
            logger.warning(
                f"LLM reranking failed: {e}, falling back to original ordering"
            )
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

        Uses JSON parser directly instead of regex for more reliable parsing.
        Falls back to 0.5 for all candidates on parse failure.
        """
        text = raw.strip()

        # Strategy 1: Parse entire response as JSON (LLM returns just the array)
        try:
            scores = json.loads(text)
            if isinstance(scores, list) and len(scores) > 0:
                return LLMReranker._clamp_scores(scores, expected_count)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract substring from first '[' to last ']'
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                candidate = text[start : end + 1]
                scores = json.loads(candidate)
                if isinstance(scores, list) and len(scores) > 0:
                    return LLMReranker._clamp_scores(scores, expected_count)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find and try each bracket pair (handles multiple arrays)
        pos = 0
        while pos < len(text):
            start = text.find("[", pos)
            if start == -1:
                break

            # Find the matching closing bracket
            end = text.find("]", start + 1)
            if end == -1:
                break

            try:
                candidate = text[start : end + 1]
                scores = json.loads(candidate)
                if isinstance(scores, list) and len(scores) > 0:
                    return LLMReranker._clamp_scores(scores, expected_count)
            except json.JSONDecodeError:
                pass

            # Move past this bracket pair
            pos = end + 1

        # Fallback: return default scores
        logger.debug(f"Failed to parse scores from LLM response: {text[:100]}")
        return [0.5] * expected_count

    @staticmethod
    def _clamp_scores(scores: list, expected_count: int) -> list[float]:
        """
        Pad/truncate scores to expected count and clamp values to [0.0, 1.0].

        Non-numeric values are replaced with 0.5.
        """
        # Pad or truncate to expected count
        if len(scores) < expected_count:
            scores = scores + [0.5] * (expected_count - len(scores))
        else:
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
