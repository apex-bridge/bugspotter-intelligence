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


def test_events_select_includes_rationale_in_correct_position():
    """Regression-guard for the events endpoint: the SELECT column list and the
    tuple unpacking that constructs each ObservabilityEvent must agree on
    column order. The `rationale` column was added between `confidence` and
    `status` — a SELECT-only or unpacking-only change would silently rotate
    every downstream column and produce wildly wrong rows.

    No real Postgres in unit CI; assert against the source string so the
    regression at least gets caught on every test run.
    """
    text = _ADMIN_SRC.read_text(encoding="utf-8")

    # SELECT list mentions rationale, sandwiched between confidence and status.
    assert (
        "confidence, rationale, status, error_kind, cached, created_at" in text
    ), "events SELECT must include rationale between confidence and status"

    # Tuple unpacking puts rationale at r[12] and bumps status/error_kind/
    # cached/created_at by one index. Pin the exact string so any column
    # rotation breaks the test before it breaks production.
    assert "confidence=r[11], rationale=r[12]," in text
    assert "status=r[13], error_kind=r[14], cached=r[15], created_at=r[16]," in text
