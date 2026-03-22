from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


def normalize_status_value(v: str | None) -> str | None:
    """Normalize status to lowercase for case-insensitive input"""
    return v.lower() if isinstance(v, str) else v


class AskRequest(BaseModel):
    """Request model for /ask endpoint"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask the AI",
        examples=["What causes null pointer exceptions?"],
    )

    context: list[str] | None = Field(
        default=None,
        description="Optional context strings (e.g., similar bug descriptions)",
        examples=[["Bug #1: App crashes on login", "Bug #2: Null pointer in auth"]],
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Randomness of response (0.0 = deterministic, 1.0 = creative)",
    )

    max_tokens: int = Field(
        default=500, ge=10, le=2000, description="Maximum length of response"
    )


class AnalyzeBugRequest(BaseModel):
    """Request model for analyzing a bug"""

    bug_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique bug identifier from main BugSpotter app",
        examples=["bug-12345"],
    )

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Bug title/summary",
        examples=["Login crashes with null pointer"],
    )

    description: str | None = Field(
        default=None, max_length=5000, description="Detailed bug description"
    )

    console_logs: list[dict] | None = Field(
        default=None, description="Browser console logs"
    )

    network_logs: list[dict] | None = Field(
        default=None, description="Network request logs"
    )

    metadata: dict | None = Field(
        default=None, description="Environment metadata (browser, OS, etc.)"
    )


class UpdateResolutionRequest(BaseModel):
    """Request model for updating bug resolution"""

    resolution: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="How the bug was fixed",
        examples=["Added null check in AuthService.java:42"],
    )

    status: str = Field(
        default="resolved",
        pattern="^(resolved|closed|wont_fix)$",
        description="New bug status",
    )

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: str | None) -> str | None:
        return normalize_status_value(v)


class APIKeyCreateBase(BaseModel):
    """Shared fields for API key creation requests."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable name for the key",
        examples=["Production API Key", "Development Key"],
    )

    rate_limit_per_minute: int = Field(
        default=60, ge=1, le=10000, description="Requests per minute limit for this key"
    )

    is_admin: bool = Field(
        default=False, description="Whether this key has admin privileges"
    )


class CreateAPIKeyRequest(APIKeyCreateBase):
    """Request model for creating a new API key"""

    tenant_id: UUID | None = Field(
        default=None,
        description="Optional. If provided, must match your authenticated tenant ID (returns 403 otherwise). The API key will always be created for your authenticated tenant.",
    )


class CreateTenantAPIKeyRequest(APIKeyCreateBase):
    """Request model for creating an API key for a specific tenant (master key endpoint).

    Unlike CreateAPIKeyRequest, this model omits tenant_id because the tenant
    is supplied via the URL path parameter, not the request body.
    """


class SearchRequest(BaseModel):
    """Request model for POST /search endpoint"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language search query",
        examples=["login page crash on mobile"],
    )

    mode: Literal["fast", "smart"] = Field(
        default="fast",
        description="Search mode: fast (vector similarity) or smart (LLM reranked)",
    )

    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results to return"
    )

    offset: int = Field(default=0, ge=0, description="Pagination offset")

    status: str | None = Field(
        default=None,
        pattern="^(open|resolved|closed|wont_fix|duplicate)$",
        description="Filter by bug status",
    )

    date_from: datetime | None = Field(
        default=None, description="Filter: bugs created on or after this date"
    )

    date_to: datetime | None = Field(
        default=None, description="Filter: bugs created on or before this date"
    )

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: str | None) -> str | None:
        return normalize_status_value(v)

    @model_validator(mode="after")
    def check_date_range(self) -> "SearchRequest":
        """Ensure date_from is not greater than date_to"""
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be less than or equal to date_to")
        return self
