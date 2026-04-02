import json
import re
from urllib.parse import urlparse
from uuid import UUID

from psycopg import AsyncConnection

from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.db.bug_repository import BugRepository
from bugspotter_intelligence.llm import LLMProvider
from bugspotter_intelligence.services.embeddings import EmbeddingProvider


class BugQueryService:
    """
    Handles bug read operations (queries)

    Queries that read state:
    - Get bug details
    - Find similar bugs
    - Get mitigation suggestions
    """

    def __init__(self, settings: Settings, llm_provider: LLMProvider, embedding_provider: EmbeddingProvider):
        self.llm = llm_provider
        self.embeddings = embedding_provider
        self.repo = BugRepository()
        self.settings = settings

    async def get_bug(
        self,
        conn: AsyncConnection,
        bug_id: str,
        tenant_id: UUID | None = None,
    ) -> dict | None:
        """
        Query: Get bug details by ID.

        Args:
            conn: Database connection
            bug_id: Bug identifier
            tenant_id: Tenant UUID for filtering

        Returns:
            Bug dict if found, None otherwise
        """
        return await self.repo.get_bug(conn, bug_id, tenant_id=tenant_id)

    async def find_similar_bugs(
        self,
        conn: AsyncConnection,
        bug_id: str,
        similarity_threshold: float | None = None,
        limit: int | None = None,
        tenant_id: UUID | None = None,
    ) -> dict:
        """
        Query: Find bugs similar to the given bug.

        Args:
            conn: Database connection
            bug_id: Bug identifier
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results
            tenant_id: Tenant UUID for filtering

        Returns:
            {
                "bug_id": str,
                "is_duplicate": bool,
                "similar_bugs": list[dict],
                "threshold_used": float
            }
        """
        # Get the bug's embedding
        bug = await self.repo.get_bug(conn, bug_id, tenant_id=tenant_id)

        if not bug:
            raise ValueError(f"Bug {bug_id} not found")

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.settings.similarity_threshold
        )
        max_bugs = limit if limit is not None else self.settings.max_similar_bugs

        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT embedding FROM bug_embeddings WHERE bug_id = %s",
                (bug_id,),
            )
            row = await cursor.fetchone()
            if not row:
                raise ValueError(f"Embedding not found for bug {bug_id}")

            embedding = row[0]

        # Find similar bugs (filtering by tenant)
        similar_bugs = await self.repo.find_similar(
            conn=conn,
            embedding=embedding,
            limit=max_bugs,
            threshold=threshold,
            tenant_id=tenant_id,
            exclude_bug_id=bug_id,
        )

        # Determine if it's a duplicate
        is_duplicate = False
        if similar_bugs and similar_bugs[0]["similarity"] >= self.settings.duplicate_threshold:
            is_duplicate = True

        return {
            "bug_id": bug_id,
            "is_duplicate": is_duplicate,
            "similar_bugs": similar_bugs,
            "threshold_used": threshold,
        }

    async def get_mitigation_suggestion(
        self,
        conn: AsyncConnection,
        bug_id: str,
        use_similar_bugs: bool = True,
        tenant_id: UUID | None = None,
    ) -> dict:
        """
        Query: Get AI-powered mitigation suggestion for a bug.

        Optionally uses similar bugs with resolutions as context.

        Args:
            conn: Database connection
            bug_id: Bug identifier
            use_similar_bugs: Whether to use similar bugs for context
            tenant_id: Tenant UUID for filtering

        Returns:
            {
                "bug_id": str,
                "mitigation_suggestion": str,
                "based_on_similar_bugs": bool
            }
        """
        # Get the bug
        bug = await self.repo.get_bug(conn, bug_id, tenant_id=tenant_id)

        if not bug:
            raise ValueError(f"Bug {bug_id} not found")

        # Get similar bugs if requested
        context = []
        if use_similar_bugs:
            similar_result = await self.find_similar_bugs(
                conn, bug_id, tenant_id=tenant_id
            )

            for similar_bug in similar_result["similar_bugs"]:
                if similar_bug.get("resolution"):
                    context.append(
                        f"Similar bug: {similar_bug['title']}\n"
                        f"Resolution: {similar_bug['resolution']}"
                    )

        # Generate mitigation
        suggestion = await self._generate_mitigation(
            title=bug["title"],
            description=bug.get("description"),
            context=context,
        )

        return {
            "bug_id": bug_id,
            "mitigation_suggestion": suggestion,
            "based_on_similar_bugs": len(context) > 0,
        }

    async def _generate_mitigation(
            self,
            title: str,
            description: str | None,
            context: list[str]
    ) -> str:
        """Generate AI mitigation suggestion"""
        prompt_parts = [f"Bug: {title}"]

        if description:
            prompt_parts.append(f"Description: {description}")

        prompt_parts.append(
            "\nProvide a concise, actionable suggestion for how to fix or mitigate this issue."
        )

        prompt = "\n".join(prompt_parts)

        suggestion = await self.llm.generate(
            prompt=prompt,
            context=context if context else None,
            temperature=0.3,
            max_tokens=300
        )

        return suggestion

    async def enrich_bug(
        self,
        bug_id: str,
        title: str,
        description: str | None = None,
        console_logs: list[dict] | None = None,
        network_logs: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """
        Analyze bug and generate enrichment data using the LLM.

        Returns category, severity, tags, root cause summary,
        affected components, and confidence scores.
        """
        # Build context from all available bug data
        context_parts = [f"Title: {title}"]

        if description:
            context_parts.append(f"Description: {description}")

        if console_logs:
            # Filter all logs for errors first, then limit (don't slice before filtering)
            error_logs = [
                log for log in console_logs
                if isinstance(log, dict)
                and str(log.get("level", "")).lower() in ("error", "warn")
            ][:10]
            if error_logs:
                log_lines = []
                for log in error_logs[:5]:
                    msg = log.get("message", str(log))
                    level = str(log.get("level", "error")).lower()
                    log_lines.append(f"[{level}] {str(msg)[:200]}")
                context_parts.append("Console errors:\n" + "\n".join(log_lines))

        if network_logs:
            # Filter all requests for failures first, then limit
            failed_requests = [
                req for req in network_logs
                if isinstance(req, dict) and (
                    (isinstance(req.get("status"), int) and req["status"] >= 400)
                    or req.get("error")
                )
            ][:10]
            if failed_requests:
                net_lines = []
                for req in failed_requests[:5]:
                    method = req.get("method", "?")
                    url = str(req.get("url", "?"))[:100]
                    status = req.get("status", "?")
                    net_lines.append(f"{method} {url} → {status}")
                context_parts.append("Failed network requests:\n" + "\n".join(net_lines))

        if metadata:
            meta_parts = []
            for key in ("browser", "os", "viewport"):
                if key in metadata:
                    meta_parts.append(f"{key}: {metadata[key]}")
            # Strip URL to path only — avoid leaking domains/query params
            if "url" in metadata:
                try:
                    page_path = urlparse(str(metadata["url"])).path or ""
                    if page_path:
                        meta_parts.append(f"page: {page_path[:100]}")
                except ValueError:
                    pass
            if meta_parts:
                context_parts.append(f"Environment: {', '.join(meta_parts)}")

        bug_context = "\n\n".join(context_parts)

        prompt = f"""Analyze this software bug report and classify it. Respond ONLY with valid JSON, no other text.

{bug_context}

Return this exact JSON structure:
{{
  "category": "<one of: ui, api, performance, authentication, database, network, crash, validation, security, other>",
  "severity": "<one of: critical, high, medium, low>",
  "root_cause": "<1-2 sentence summary of the likely root cause>",
  "components": ["<affected component names, e.g. CheckoutForm, PaymentService>"],
  "tags": ["<descriptive tags, e.g. null-pointer, race-condition, timeout>"]
}}"""

        raw_response = await self.llm.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=400,
        )

        return self._parse_enrichment_response(bug_id, raw_response)

    def _parse_enrichment_response(self, bug_id: str, raw: str) -> dict:
        """Parse LLM JSON response into enrichment format with confidence scores."""
        # Try direct parse first, then extract from markdown
        parsed = None
        try:
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        if parsed is None:
            # Extract JSON blocks — take the last one (LLM may echo the template first)
            json_blocks = re.findall(
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL
            )
            for candidate in reversed(json_blocks):
                try:
                    parsed = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue

        if not isinstance(parsed, dict):
            return self._default_enrichment(bug_id)

        # Validate and normalize fields
        valid_categories = {
            "ui", "api", "performance", "authentication", "database",
            "network", "crash", "validation", "security", "other",
        }
        valid_severities = {"critical", "high", "medium", "low"}

        category = str(parsed.get("category", "other")).lower()
        if category not in valid_categories:
            category = "other"

        severity = str(parsed.get("severity", "medium")).lower()
        if severity not in valid_severities:
            severity = "medium"

        raw_root_cause = parsed.get("root_cause")
        if isinstance(raw_root_cause, str):
            root_cause = raw_root_cause.strip()[:500]
        else:
            root_cause = ""

        components = []
        for c in (parsed.get("components") or [])[:10]:
            if not isinstance(c, str):
                continue
            cleaned = c.strip()
            if cleaned and not cleaned.startswith("<"):
                components.append(cleaned[:100])

        tags = []
        for t in (parsed.get("tags") or [])[:10]:
            if not isinstance(t, str):
                continue
            cleaned = t.strip()
            if cleaned and not cleaned.startswith("<"):
                tags.append(cleaned[:50])

        # Confidence: higher when LLM provided real (non-placeholder) output
        has_root_cause = bool(root_cause) and not root_cause.startswith("<")
        base_confidence = 0.75 if has_root_cause else 0.5

        if not has_root_cause:
            root_cause = "Unable to determine root cause"

        return {
            "bug_id": bug_id,
            "category": category,
            "suggested_severity": severity,
            "tags": tags,
            "root_cause_summary": root_cause,
            "affected_components": components,
            "confidence": {
                "category": base_confidence,
                "severity": base_confidence,
                "tags": base_confidence * 0.9,
                "root_cause": base_confidence if has_root_cause else 0.3,
                "components": base_confidence * 0.85 if components else 0.2,
            },
        }

    def _default_enrichment(self, bug_id: str) -> dict:
        """Return default enrichment when LLM parsing fails."""
        return {
            "bug_id": bug_id,
            "category": "other",
            "suggested_severity": "medium",
            "tags": [],
            "root_cause_summary": "Unable to determine root cause from available data",
            "affected_components": [],
            "confidence": {
                "category": 0.2,
                "severity": 0.2,
                "tags": 0.1,
                "root_cause": 0.1,
                "components": 0.1,
            },
        }