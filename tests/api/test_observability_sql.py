"""Regression test for percentile_cont casts in observability admin routes.

PostgreSQL's `percentile_cont(fraction) WITHIN GROUP (ORDER BY expr)` requires
`expr` to be `double precision` / `numeric`; `latency_ms` is INTEGER, so without
a cast the query fails at runtime with:
    function percentile_cont(double precision) ordered by type integer does not exist

A real-DB integration test would catch this but we don't run Postgres in unit
CI; assert the cast exists in the source string instead so any regression that
strips it gets caught here.
"""

from pathlib import Path

_ADMIN_SRC = (
    Path(__file__).resolve().parents[2]
    / "src" / "bugspotter_intelligence" / "api" / "routes" / "admin.py"
)


def test_percentile_cont_casts_latency_ms_to_float():
    text = _ADMIN_SRC.read_text(encoding="utf-8")
    correct = "ORDER BY latency_ms::float"
    wrong = "ORDER BY latency_ms)"
    # At least one correctly-cast occurrence per percentile call (summary has 2,
    # by-operation has 2 → 4 total).
    assert text.count(correct) == 4, (
        f"expected 4 occurrences of '{correct}' in admin.py, found {text.count(correct)}"
    )
    # Naked latency_ms inside an ORDER BY would have a closing paren right after.
    # This catches a regression where the cast moves to the wrong side of the
    # paren (e.g. `ORDER BY latency_ms)::float` which is valid SQL but a NOP cast).
    assert wrong not in text, f"found leftover uncast ORDER BY latency_ms): {wrong!r}"


def test_summary_has_by_day_cost_and_token_rollup():
    """The cost dashboard relies on a per-day rollup of cost + tokens. Assert the
    aggregation is present (real-DB behavior is covered by integration tests)."""
    text = _ADMIN_SRC.read_text(encoding="utf-8")
    assert "date_trunc('day', created_at)::date AS day" in text
    assert "GROUP BY day" in text
    # Tokens are the always-populated metric; cost is the derived $ estimate.
    for col in ("SUM(tokens_in)", "SUM(tokens_out)", "SUM(cost_micros_usd)"):
        assert col in text, f"by_day rollup missing {col}"
