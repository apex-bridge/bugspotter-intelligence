"""NL → DedupRule parser endpoint.

POST /rules/parse-nl
  body:  ParseNLRuleRequest  — the user's NL string + tenant context
  resp:  ParseNLRuleResponse — structured DedupRule draft (or null + errors)

Used by the BugSpotter admin UI to let operators describe a dedup automation
in plain language. The endpoint validates against the DedupRule schema and
returns a draft for the admin to review before saving.
"""

from fastapi import APIRouter, Depends

from bugspotter_intelligence.api.deps import get_llm_provider, get_settings
from bugspotter_intelligence.auth import TenantContext
from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.llm import LLMProvider
from bugspotter_intelligence.models.requests import ParseNLRuleRequest
from bugspotter_intelligence.models.responses import ParseNLRuleResponse
from bugspotter_intelligence.rate_limiting import check_rate_limit
from bugspotter_intelligence.services.rule_parser_service import RuleParserService

router = APIRouter(prefix="/rules", tags=["Rules"])


@router.post("/parse-nl", response_model=ParseNLRuleResponse)
async def parse_nl_rule(
    body: ParseNLRuleRequest,
    tenant: TenantContext = Depends(check_rate_limit),
    provider: LLMProvider = Depends(get_llm_provider),
    settings: Settings = Depends(get_settings),
) -> ParseNLRuleResponse:
    """Convert a natural-language rule description into a structured DedupRule.

    The response always includes the LLM model used so the admin UI can
    surface it. `raw_llm_output` is included for debugging — the UI should
    not display it to end users by default.
    """
    service = RuleParserService(provider)
    result = await service.parse_nl_to_rule(
        nl=body.nl,
        available_integrations=body.available_integrations,
        available_slack_channels=body.available_slack_channels,
        available_email_templates=body.available_email_templates,
    )

    # `settings.{provider}_model` is the model identifier (e.g.
    # "llama3.2:3b"); falling back to the provider name ("ollama") would
    # silently put the wrong value into the response. Use a sentinel
    # instead so consumers can detect the misconfiguration.
    model = getattr(settings, f"{settings.llm_provider}_model", "unknown")
    return ParseNLRuleResponse(
        draft=result.draft,
        errors=result.errors,
        clarifications=result.clarifications,
        raw_llm_output=result.raw_llm_output,
        model=model,
    )
