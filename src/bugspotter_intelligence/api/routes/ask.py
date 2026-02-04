"""Q&A endpoint using LLM"""

from fastapi import APIRouter, Depends
from bugspotter_intelligence.api.deps import get_llm_provider, get_settings
from bugspotter_intelligence.auth import TenantContext
from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.llm import LLMProvider
from bugspotter_intelligence.models import AskRequest, AskResponse
from bugspotter_intelligence.rate_limiting import check_rate_limit

router = APIRouter(prefix="/ask", tags=["Q&A"])


@router.post("", response_model=AskResponse)
async def ask_question(
        body: AskRequest,
        tenant: TenantContext = Depends(check_rate_limit),
        provider: LLMProvider = Depends(get_llm_provider),
        settings: Settings = Depends(get_settings)
) -> AskResponse:
    """
    Ask a question to the AI

    Similar to:
    - Spring: @PostMapping("/ask")
    - ASP.NET: [HttpPost("ask")]
    """
    answer = await provider.generate(
        prompt=body.question,
        context=body.context,
        temperature=body.temperature,
        max_tokens=body.max_tokens
    )

    return AskResponse(
        answer=answer,
        provider=settings.llm_provider,
        model=getattr(settings, f"{settings.llm_provider}_model")
    )
