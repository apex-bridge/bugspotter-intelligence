from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""

    answer: str = Field(..., description="AI-generated answer")

    provider: str = Field(
        ..., description="LLM provider used (e.g., 'ollama', 'claude')"
    )

    model: str = Field(..., description="Model used (e.g., 'llama3.1:8b')")


class SimilarBug(BaseModel):
    """Model for a similar bug in search results"""

    bug_id: str
    title: str
    description: Optional[str] = None
    status: str
    resolution: Optional[str] = None
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")


class AnalyzeBugResponse(BaseModel):
    """Response model for bug analysis"""

    bug_id: str
    embedding_generated: bool
    stored: bool = True


class SimilarBugsResponse(BaseModel):
    """Response model for similar bugs query"""

    bug_id: str
    is_duplicate: bool
    similar_bugs: list[SimilarBug]
    threshold_used: float


class MitigationResponse(BaseModel):
    """Response model for mitigation suggestion"""

    bug_id: str
    mitigation_suggestion: str
    based_on_similar_bugs: bool


class BugDetailResponse(BaseModel):
    """Response model for bug details"""

    bug_id: str
    title: str
    description: Optional[str] = None
    status: str
    resolution: Optional[str] = None
    resolution_summary: Optional[str] = None
    created_at: str
    updated_at: str


class ResolutionUpdateResponse(BaseModel):
    """Response model for resolution update"""

    bug_id: str
    status: str
    resolution_summary: str
    updated: bool = True


class APIKeyResponse(BaseModel):
    """API key response (masked - key_hash not included)"""

    id: UUID
    tenant_id: UUID
    key_prefix: str = Field(..., description="First 12 chars of key for identification")
    name: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool
    rate_limit_per_minute: int
    is_admin: bool

    model_config = {"from_attributes": True}


class CreateAPIKeyResponse(BaseModel):
    """Response after creating API key (includes plain key ONCE)"""

    api_key: APIKeyResponse
    plain_key: str = Field(..., description="The full API key - store securely!")
    warning: str = "Store this key securely. It will not be shown again."


class APIKeyListResponse(BaseModel):
    """List of API keys for a tenant"""

    keys: list[APIKeyResponse]
    total: int


class SearchResult(BaseModel):
    """Single result in a search response"""

    bug_id: str
    title: str
    description: Optional[str] = None
    status: str
    resolution: Optional[str] = None
    similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity score"
    )
    created_at: datetime

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_created_at(cls, v: datetime | str) -> datetime:
        """Parse ISO string to datetime if needed (for cached results)"""
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v


class SearchResponse(BaseModel):
    """Response model for POST /search endpoint"""

    results: list[SearchResult]
    total: int = Field(
        ..., ge=0, description="Total matching results (before pagination)"
    )
    limit: int
    offset: int
    mode: Literal["fast", "smart"]
    query: str
    cached: bool = False


class CacheStatsResponse(BaseModel):
    """Response model for GET /admin/cache/stats"""

    available: bool = Field(..., description="Whether Redis is available")
    keyspace_hits: int = Field(
        ..., ge=0, description="Total cache hits since Redis started"
    )
    keyspace_misses: int = Field(
        ..., ge=0, description="Total cache misses since Redis started"
    )
    hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate (0.0-1.0)")
