import logging
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from psycopg import AsyncConnection

from bugspotter_intelligence.db.bug_repository import BugRepository
from bugspotter_intelligence.llm import LLMProvider
from bugspotter_intelligence.services.embeddings import EmbeddingProvider
from bugspotter_intelligence.utils.log_extractor import build_embedding_text

if TYPE_CHECKING:
    from bugspotter_intelligence.cache.service import CacheService

logger = logging.getLogger(__name__)


class BugCommandService:
    """
    Handles bug write operations (commands)

    Commands that change state:
    - Analyze and store new bugs
    - Update bug resolutions
    - Mark bugs as duplicates
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        cache: Optional["CacheService"] = None,
    ):
        self.llm = llm_provider
        self.embeddings = embedding_provider
        self.repo = BugRepository()
        self.cache = cache

    async def analyze_and_store_bug(
        self,
        conn: AsyncConnection,
        bug_id: str,
        title: str,
        description: Optional[str] = None,
        console_logs: Optional[list[dict]] = None,
        network_logs: Optional[list[dict]] = None,
        metadata: Optional[dict] = None,
        tenant_id: Optional[UUID] = None,
    ) -> dict:
        """
        Command: Analyze bug and store its embedding.

        Returns analysis result without querying for similar bugs
        (Query service handles that).

        Args:
            conn: Database connection
            bug_id: Unique bug identifier
            title: Bug title
            description: Bug description
            console_logs: Console log entries
            network_logs: Network request logs
            metadata: Additional metadata
            tenant_id: Tenant UUID for multi-tenancy

        Returns:
            {
                "bug_id": str,
                "embedding_generated": bool,
                "embedding_text": str
            }
        """
        # Build text for embedding
        embedding_text = build_embedding_text(
            title=title,
            description=description,
            console_logs=console_logs,
            network_logs=network_logs,
            metadata=metadata,
        )

        # Generate embedding with DedupKit
        embedding = self.embeddings.embed(embedding_text)

        # Store in database
        await self.repo.insert_bug(
            conn=conn,
            bug_id=bug_id,
            title=title,
            description=description,
            embedding=embedding,
            tenant_id=tenant_id,
        )

        # Invalidate cached search results for this tenant
        if self.cache is not None and tenant_id is not None:
            await self.cache.invalidate_tenant(tenant_id)

        return {
            "bug_id": bug_id,
            "embedding_generated": True,
            "embedding_text": embedding_text[:200] + "...",  # Truncate for response
        }

    async def update_bug_resolution(
        self,
        conn: AsyncConnection,
        bug_id: str,
        resolution: str,
        status: str = "resolved",
        tenant_id: Optional[UUID] = None,
    ) -> dict:
        """
        Command: Update bug with resolution information.

        This is called when a bug is fixed in the main BugSpotter app.

        Args:
            conn: Database connection
            bug_id: Bug identifier
            resolution: Resolution description
            status: New status (default: "resolved")
            tenant_id: Tenant UUID for ownership verification

        Returns:
            {
                "bug_id": str,
                "status": str,
                "resolution_summary": str,
                "updated": bool
            }
        """
        # Optionally: Generate AI summary of the resolution
        resolution_summary = await self._generate_resolution_summary(resolution)

        # Update in database
        updated = await self.repo.update_resolution(
            conn=conn,
            bug_id=bug_id,
            resolution=resolution,
            resolution_summary=resolution_summary,
            status=status,
            tenant_id=tenant_id,
        )

        return {
            "bug_id": bug_id,
            "status": status,
            "resolution_summary": resolution_summary,
            "updated": updated,
        }

    async def _generate_resolution_summary(self, resolution: str) -> str:
        """Generate a concise summary of the resolution for future reference"""
        prompt = (
            f"Summarize this bug resolution in one concise sentence:\n\n"
            f"{resolution}\n\n"
            f"Summary:"
        )

        summary = await self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=100
        )

        return summary.strip()