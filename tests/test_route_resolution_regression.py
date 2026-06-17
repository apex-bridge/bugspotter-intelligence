"""Regression guard for the #43 production incident.

FastAPI 0.137.0 rewrote ``include_router`` so ``app.routes`` is a tree of
intermediate ``_IncludedRouter`` objects instead of a flat list of ``APIRoute``.
``prometheus-fastapi-instrumentator`` iterates ``app.routes`` and reads
``route.path`` for every request, which ``_IncludedRouter`` doesn't have — so
on 0.137+ *every* ``/api/v1/*`` route 500'd with::

    AttributeError: '_IncludedRouter' object has no attribute 'path'

We capped ``fastapi<0.137`` (#43) and locked the full closure (#44). These tests
fail loudly in CI — instead of in prod — if a future bump (FastAPI or the
instrumentator) reintroduces the incompatibility, regardless of how it happens.
"""

import fastapi
import pytest
from packaging.version import Version
from prometheus_fastapi_instrumentator import routing
from starlette.requests import Request

from bugspotter_intelligence.main import create_app


@pytest.fixture(scope="module")
def app() -> fastapi.FastAPI:
    """The real application, built once and shared across the cases below."""
    return create_app()


def _make_request(app: fastapi.FastAPI, path: str) -> Request:
    """Minimal ASGI request carrying the real app, enough for route resolution."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "raw_path": path.encode(),
            "headers": [],
            "query_string": b"",
            "app": app,
        }
    )


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/admin/status",
        "/api/v1/admin/observability/summary",
    ],
)
def test_instrumentator_resolves_route_name_without_crashing(app: fastapi.FastAPI, path: str):
    """The Prometheus instrumentator must resolve names over the real route tree.

    This runs the exact code path that 500'd in prod (#43). If FastAPI's routing
    model and the instrumentator are ever incompatible again, get_route_name
    raises AttributeError here rather than taking the API down.
    """
    request = _make_request(app, path)

    # Raises AttributeError('_IncludedRouter' ... 'path') if the regression returns.
    name = routing.get_route_name(request)

    assert name == path


def test_fastapi_capped_below_incompatible_routing():
    """Belt-and-suspenders: document and enforce the <0.137 cap from #43."""
    assert Version(fastapi.__version__) < Version("0.137"), (
        f"fastapi {fastapi.__version__} >= 0.137 reintroduces the _IncludedRouter "
        "route tree that prometheus-fastapi-instrumentator can't resolve (see #43). "
        "Keep the cap until an instrumentator release supports it."
    )
