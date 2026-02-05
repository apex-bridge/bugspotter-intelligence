"""Tests for request model validation"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from bugspotter_intelligence.models.requests import SearchRequest


class TestSearchRequest:
    """Tests for SearchRequest model validation"""

    def test_valid_date_range(self):
        """Test that valid date ranges are accepted"""
        request = SearchRequest(
            query="test query",
            date_from=datetime(2024, 1, 1),
            date_to=datetime(2024, 12, 31),
        )
        assert request.date_from < request.date_to

    def test_equal_dates_valid(self):
        """Test that equal dates are valid"""
        same_date = datetime(2024, 6, 15)
        request = SearchRequest(
            query="test query", date_from=same_date, date_to=same_date
        )
        assert request.date_from == request.date_to

    def test_invalid_date_range_raises_error(self):
        """Test that date_from > date_to raises ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(
                query="test query",
                date_from=datetime(2024, 12, 31),
                date_to=datetime(2024, 1, 1),
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "date_from must be less than or equal to date_to" in str(
            errors[0]["ctx"]["error"]
        )

    def test_none_dates_accepted(self):
        """Test that None dates don't trigger validation"""
        # Both None
        request1 = SearchRequest(query="test query", date_from=None, date_to=None)
        assert request1.date_from is None
        assert request1.date_to is None

        # Only date_from
        request2 = SearchRequest(
            query="test query", date_from=datetime(2024, 1, 1), date_to=None
        )
        assert request2.date_from is not None
        assert request2.date_to is None

        # Only date_to
        request3 = SearchRequest(
            query="test query", date_from=None, date_to=datetime(2024, 12, 31)
        )
        assert request3.date_from is None
        assert request3.date_to is not None
