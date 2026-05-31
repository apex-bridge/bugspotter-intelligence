"""Caller-facing endpoints for AI observability (feedback, confidence)."""

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg import AsyncConnection
from psycopg.errors import DataError, IntegrityError

from bugspotter_intelligence.api.deps import get_db_connection
from bugspotter_intelligence.auth import TenantContext
from bugspotter_intelligence.models.requests import SubmitFeedbackRequest
from bugspotter_intelligence.models.responses import SubmitFeedbackResponse
from bugspotter_intelligence.rate_limiting import check_rate_limit

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

# Sentinel used in place of NULL for anonymous feedback so the (event_id, user_ref)
# UNIQUE constraint actually dedupes — Postgres treats NULLs as distinct.
_ANON_USER_REF = "<ANON>"


@router.post("/feedback", response_model=SubmitFeedbackResponse, status_code=201)
async def submit_feedback(
    body: SubmitFeedbackRequest,
    tenant: TenantContext = Depends(check_rate_limit),
    conn: AsyncConnection = Depends(get_db_connection),
) -> SubmitFeedbackResponse:
    """Record a user verdict on a prior intelligence_event.

    The event must belong to the caller's tenant; cross-tenant feedback is rejected
    with 404 (intentionally indistinguishable from "event doesn't exist").
    """
    user_ref = body.user_ref or _ANON_USER_REF

    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT tenant_id FROM intelligence_event WHERE id = %s",
            (body.event_id,),
        )
        row = await cur.fetchone()
        if row is None or row[0] != tenant.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="event_id not found",
            )

        try:
            await cur.execute(
                """
                INSERT INTO intelligence_feedback (event_id, tenant_id, verdict, user_ref, note)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (event_id, user_ref) DO UPDATE
                  SET verdict = EXCLUDED.verdict, note = EXCLUDED.note
                RETURNING id
                """,
                (body.event_id, tenant.tenant_id, body.verdict, user_ref, body.note),
            )
        except (IntegrityError, DataError) as exc:
            # Validation-class write rejections only (constraint / type / range).
            # Operational errors (connection, transaction abort) fall through to
            # the global 500 handler so we don't mask outages as client errors.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not record feedback",
            ) from exc

        feedback_row = await cur.fetchone()
        await conn.commit()

    return SubmitFeedbackResponse(feedback_id=feedback_row[0])
