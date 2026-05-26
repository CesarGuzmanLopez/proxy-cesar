"""GET /v1/models — List available pseudo-models + local models.

sprint §10 — optimistic capability advertising to prevent silent content stripping.
analisis.md: all capabilities advertised as true for all pseudo-models.

Local models (Ollama, LM Studio) are discovered live and included with
their actual capabilities and a conservative 30% context window limit.
"""

from fastapi import APIRouter, Request

from src.adapters.local_discovery import discover_local_models

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
    """Return all pseudo-models + discovered local models.

    Pseudo-models advertise optimistic capabilities.
    Local models report their actual capabilities from the provider.
    """
    config = request.app.state.config
    models = []

    # Pseudo-models from config
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

    # Local models from Ollama/LM Studio (discovered live)
    local_models = await discover_local_models()
    for loc in local_models:
        models.append({
            "id": loc.id,
            "object": "model",
            "created": 1700000000,
            "owned_by": loc.provider,
            "display_name": loc.display_name,
            "description": (
                f"Local model via {loc.provider}. "
                f"{loc.parameter_size or ''} {loc.architecture or ''}. "
                f"Context: {loc.context_window} tokens (30% of reported)."
            ),
            "capabilities": {
                "vision": loc.vision,
                "tools": loc.tools,
                "parallel_tools": False,
                "streaming": True,
                "function_calling": False,
            },
            "context_window": loc.context_window,
            "input_token_threshold": loc.context_window,
            "pricing": {
                "estimated_input_cost_per_1k": 0,
                "estimated_output_cost_per_1k": 0,
            },
        })

    return {"object": "list", "data": models}
