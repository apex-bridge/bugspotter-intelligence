"""Tests for LLMReranker"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bugspotter_intelligence.services.reranker import LLMReranker


@pytest.fixture
def mock_llm():
    """Mock LLM provider for reranker"""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value="[0.9, 0.3, 0.7]")
    return provider


@pytest.fixture
def reranker(mock_llm):
    """Create reranker with mock LLM"""
    return LLMReranker(mock_llm, timeout_seconds=5.0)


@pytest.fixture
def sample_candidates():
    """Sample candidates from fast search"""
    return [
        {
            "bug_id": "bug-001",
            "title": "Login page crash on mobile",
            "description": "App crashes when tapping login button",
            "status": "open",
            "resolution": None,
            "similarity": 0.85,
        },
        {
            "bug_id": "bug-002",
            "title": "Password reset email not sent",
            "description": "Users don't receive reset emails",
            "status": "resolved",
            "resolution": "Fixed SMTP config",
            "similarity": 0.72,
        },
        {
            "bug_id": "bug-003",
            "title": "Auth token expires too early",
            "description": "Session timeout after 5 minutes",
            "status": "open",
            "resolution": None,
            "similarity": 0.68,
        },
    ]


class TestRerank:
    """Tests for LLMReranker.rerank"""

    @pytest.mark.asyncio
    async def test_reranks_by_llm_scores(self, reranker, mock_llm, sample_candidates):
        """Should reorder candidates based on LLM scores"""
        # LLM scores: [0.9, 0.3, 0.7] → order should be bug-001, bug-003, bug-002
        mock_llm.generate = AsyncMock(return_value="[0.9, 0.3, 0.7]")

        results, llm_used = await reranker.rerank(
            "login crash", sample_candidates, return_limit=3
        )

        assert llm_used is True
        assert len(results) == 3
        assert results[0]["bug_id"] == "bug-001"  # score 0.9
        assert results[1]["bug_id"] == "bug-003"  # score 0.7
        assert results[2]["bug_id"] == "bug-002"  # score 0.3

    @pytest.mark.asyncio
    async def test_updates_similarity_with_llm_scores(
        self, reranker, mock_llm, sample_candidates
    ):
        """Should replace similarity scores with LLM-assigned scores"""
        mock_llm.generate = AsyncMock(return_value="[0.95, 0.2, 0.6]")

        results, _ = await reranker.rerank(
            "login crash", sample_candidates, return_limit=3
        )

        assert results[0]["similarity"] == 0.95
        assert results[1]["similarity"] == 0.6
        assert results[2]["similarity"] == 0.2

    @pytest.mark.asyncio
    async def test_respects_return_limit(self, reranker, mock_llm, sample_candidates):
        """Should return only return_limit results"""
        mock_llm.generate = AsyncMock(return_value="[0.9, 0.3, 0.7]")

        results, llm_used = await reranker.rerank(
            "login crash", sample_candidates, return_limit=2
        )

        assert llm_used is True
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_falls_back_on_timeout(self, mock_llm, sample_candidates):
        """Should fall back to original ordering on LLM timeout"""

        async def slow_generate(**kwargs):
            await asyncio.sleep(10)
            return "[0.9, 0.3, 0.7]"

        mock_llm.generate = slow_generate
        reranker = LLMReranker(mock_llm, timeout_seconds=0.1)

        results, llm_used = await reranker.rerank(
            "login crash", sample_candidates, return_limit=3
        )

        assert llm_used is False
        # Should return original order (first 3)
        assert results[0]["bug_id"] == "bug-001"
        assert results[1]["bug_id"] == "bug-002"
        assert results[2]["bug_id"] == "bug-003"

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_exception(
        self, reranker, mock_llm, sample_candidates
    ):
        """Should fall back to original ordering on LLM error"""
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM unavailable"))

        results, llm_used = await reranker.rerank(
            "login crash", sample_candidates, return_limit=3
        )

        assert llm_used is False
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_empty_candidates(self, reranker):
        """Should handle empty candidates list"""
        results, llm_used = await reranker.rerank("query", [], return_limit=5)

        assert results == []
        assert llm_used is True

    @pytest.mark.asyncio
    async def test_prompt_includes_query(self, reranker, mock_llm, sample_candidates):
        """Should include the search query in the LLM prompt"""
        mock_llm.generate = AsyncMock(return_value="[0.5, 0.5, 0.5]")

        await reranker.rerank("login crash", sample_candidates, return_limit=3)

        call_kwargs = mock_llm.generate.call_args.kwargs
        assert "login crash" in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_prompt_includes_candidate_titles(
        self, reranker, mock_llm, sample_candidates
    ):
        """Should include candidate titles in the prompt"""
        mock_llm.generate = AsyncMock(return_value="[0.5, 0.5, 0.5]")

        await reranker.rerank("query", sample_candidates, return_limit=3)

        prompt = mock_llm.generate.call_args.kwargs["prompt"]
        assert "Login page crash on mobile" in prompt
        assert "Password reset email not sent" in prompt
        assert "Auth token expires too early" in prompt

    @pytest.mark.asyncio
    async def test_uses_zero_temperature(self, reranker, mock_llm, sample_candidates):
        """Should use temperature=0.0 for deterministic scoring"""
        mock_llm.generate = AsyncMock(return_value="[0.5, 0.5, 0.5]")

        await reranker.rerank("query", sample_candidates, return_limit=3)

        call_kwargs = mock_llm.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0


class TestParseScores:
    """Tests for LLMReranker._parse_scores"""

    def test_valid_json_array(self):
        """Should parse valid JSON array of floats"""
        scores = LLMReranker._parse_scores("[0.9, 0.3, 0.7]", 3)
        assert scores == [0.9, 0.3, 0.7]

    def test_json_with_surrounding_text(self):
        """Should extract JSON array from surrounding text"""
        scores = LLMReranker._parse_scores(
            "Here are the scores: [0.8, 0.2, 0.5] based on relevance", 3
        )
        assert scores == [0.8, 0.2, 0.5]

    def test_clamps_values_above_one(self):
        """Should clamp scores above 1.0 to 1.0"""
        scores = LLMReranker._parse_scores("[1.5, 0.5, 2.0]", 3)
        assert scores == [1.0, 0.5, 1.0]

    def test_clamps_values_below_zero(self):
        """Should clamp scores below 0.0 to 0.0"""
        scores = LLMReranker._parse_scores("[-0.5, 0.5, -1.0]", 3)
        assert scores == [0.0, 0.5, 0.0]

    def test_falls_back_to_half_on_no_json(self):
        """Should return 0.5 for all when no JSON array found"""
        scores = LLMReranker._parse_scores("I can't score these", 3)
        assert scores == [0.5, 0.5, 0.5]

    def test_falls_back_to_half_on_invalid_json(self):
        """Should return 0.5 for all on malformed JSON"""
        scores = LLMReranker._parse_scores("[0.9, invalid, 0.7]", 3)
        assert scores == [0.5, 0.5, 0.5]

    def test_pads_short_array(self):
        """Should pad with 0.5 if fewer scores than expected"""
        scores = LLMReranker._parse_scores("[0.9, 0.3]", 4)
        assert scores == [0.9, 0.3, 0.5, 0.5]

    def test_truncates_long_array(self):
        """Should truncate if more scores than expected"""
        scores = LLMReranker._parse_scores("[0.9, 0.3, 0.7, 0.1, 0.5]", 3)
        assert scores == [0.9, 0.3, 0.7]

    def test_handles_non_numeric_values(self):
        """Should use 0.5 for non-numeric values in array"""
        scores = LLMReranker._parse_scores('[0.9, "high", 0.7]', 3)
        assert scores == [0.9, 0.5, 0.7]

    def test_handles_integer_scores(self):
        """Should handle integer values (0 and 1)"""
        scores = LLMReranker._parse_scores("[1, 0, 1]", 3)
        assert scores == [1.0, 0.0, 1.0]


class TestBuildPrompt:
    """Tests for LLMReranker._build_prompt"""

    def test_includes_status(self, reranker, sample_candidates):
        """Should include bug status in prompt"""
        prompt = reranker._build_prompt("query", sample_candidates)
        assert "[open]" in prompt
        assert "[resolved]" in prompt

    def test_includes_resolution_when_present(self, reranker, sample_candidates):
        """Should include resolution for resolved bugs"""
        prompt = reranker._build_prompt("query", sample_candidates)
        assert "Fixed SMTP config" in prompt

    def test_truncates_long_descriptions(self, reranker):
        """Should truncate descriptions longer than 200 chars and add ellipsis"""
        candidates = [
            {
                "title": "Bug",
                "description": "x" * 500,
                "status": "open",
                "resolution": None,
            }
        ]
        prompt = reranker._build_prompt("query", candidates)
        # Description should be truncated to 200 chars with ellipsis
        assert ("x" * 200 + "...") in prompt
        assert "x" * 201 not in prompt

    def test_truncates_long_resolutions(self, reranker):
        """Should truncate resolutions longer than 100 chars and add ellipsis"""
        candidates = [
            {
                "title": "Bug",
                "description": "Short description",
                "status": "resolved",
                "resolution": "y" * 200,
            }
        ]
        prompt = reranker._build_prompt("query", candidates)
        # Resolution should be truncated to 100 chars with ellipsis
        assert ("y" * 100 + "...") in prompt
        assert "y" * 101 not in prompt
