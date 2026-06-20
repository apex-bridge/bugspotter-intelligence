"""Per-million-token USD prices, in micro-dollars. None = unknown (NOT zero)."""

# (provider, model_prefix) -> (input_micros_per_mtok, output_micros_per_mtok)
# Prefix-match: model 'claude-sonnet-4-6-20260301' matches prefix 'claude-sonnet-4-6'.
#
# The provider key MUST be the name recorder._resolve_provider_name() emits —
# the class name minus "Provider", lowercased: ClaudeProvider -> "claude",
# OpenAIProvider -> "openai". Keying Anthropic under "anthropic" (the API-vendor
# name) silently never matched, so every Claude call recorded cost_micros=None.
_PRICES_PER_MTOK: dict[tuple[str, str], tuple[int, int]] = {
    ("claude",  "claude-opus-4-7"):   (15_000_000, 75_000_000),
    ("claude",  "claude-sonnet-4-6"): (3_000_000, 15_000_000),
    ("claude",  "claude-haiku-4-5"):  (1_000_000, 5_000_000),
    ("openai",  "gpt-4o-mini"):       (150_000, 600_000),
    ("openai",  "gpt-4o"):            (2_500_000, 10_000_000),
}


def price_micros(
    provider: str,
    model: str,
    tokens_in: int | None,
    tokens_out: int | None,
) -> int | None:
    if tokens_in is None and tokens_out is None:
        return None
    p_in = p_out = None
    for (prov, prefix), (mi, mo) in _PRICES_PER_MTOK.items():
        if prov == provider and model.startswith(prefix):
            p_in, p_out = mi, mo
            break
    if p_in is None:
        return None
    # round() not //: integer division on cheap models (e.g. 5 tokens × $0.15/Mtok)
    # truncates to 0 micros and systematically under-counts cost over volume.
    cost = 0
    if tokens_in:
        cost += int(round((tokens_in * p_in) / 1_000_000))
    if tokens_out:
        cost += int(round((tokens_out * p_out) / 1_000_000))
    return cost
