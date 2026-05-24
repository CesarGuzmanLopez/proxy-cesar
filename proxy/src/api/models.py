"""GET /v1/models — List available pseudo-models.

sprint §10 — optimistic capability advertising to prevent silent content stripping.
analisis.md: all capabilities advertised as true for all pseudo-models.
"""

from fastapi import APIRouter, Request

router = APIRouter()

# Optimistic capabilities — always true for every pseudo-model
# Prevents clients (Continue, LibreChat, etc.) from silently stripping content
ALL_CAPABILITIES = {
    "vision": True,
    "tools": True,
    "parallel_tools": True,
    "streaming": True,
    "function_calling": True,
}


@router.get("/v1/models")
async def list_models(request: Request):
    """Return all pseudo-models with optimistic capabilities.

    Every model advertises the same capabilities block.
    Actual validation happens at request time (Sprint 2+).
    """
    config = request.app.state.config
    models = []
    for name, pm in config.pseudo_models.items():
        models.append({
            "id": name,
            "object": "model",
            "created": 1700000000,
            "owned_by": "proxy-cesar",
            "display_name": pm.display_name,
            "description": pm.description,
            "capabilities": dict(ALL_CAPABILITIES),
            "context_window": pm.context_window,
            "input_token_threshold": pm.input_token_threshold,
            "pricing": {
                "estimated_input_cost_per_1k": None,
                "estimated_output_cost_per_1k": None,
            },
        })
    return {"object": "list", "data": models}
