from uuid import UUID

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for /ask endpoint"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask the AI",
        examples=["What causes null pointer exceptions?"]
    )

    context: list[str] | None = Field(
        default=None,
        description="Optional context strings (e.g., similar bug descriptions)",
        examples=[["Bug #1: App crashes on login", "Bug #2: Null pointer in auth"]]
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Randomness of response (0.0 = deterministic, 1.0 = creative)"
    )

    max_tokens: int = Field(
        default=500,
        ge=10,
        le=2000,
        description="Maximum length of response"
    )


class AnalyzeBugRequest(BaseModel):
    """Request model for analyzing a bug"""

    bug_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique bug identifier from main BugSpotter app",
        examples=["bug-12345"]
    )

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Bug title/summary",
        examples=["Login crashes with null pointer"]
    )

    description: str | None = Field(
        default=None,
        max_length=5000,
        description="Detailed bug description"
    )

    console_logs: list[dict] | None = Field(
        default=None,
        description="Browser console logs"
    )

    network_logs: list[dict] | None = Field(
        default=None,
        description="Network request logs"
    )

    metadata: dict | None = Field(
        default=None,
        description="Environment metadata (browser, OS, etc.)"
    )


class UpdateResolutionRequest(BaseModel):
    """Request model for updating bug resolution"""

    resolution: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="How the bug was fixed",
        examples=["Added null check in AuthService.java:42"]
    )

    status: str = Field(
        default="resolved",
        pattern="^(resolved|closed|wont_fix)$",
        description="New bug status"
    )


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating a new API key"""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable name for the key",
        examples=["Production API Key", "Development Key"]
    )

    tenant_id: UUID | None = Field(
        default=None,
        description="Tenant ID (reserved for future use; currently must match requesting admin's tenant)"
    )

    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="Requests per minute limit for this key"
    )

    is_admin: bool = Field(
        default=False,
        description="Whether this key has admin privileges"
    )
