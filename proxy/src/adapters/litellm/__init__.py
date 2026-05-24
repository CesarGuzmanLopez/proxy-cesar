"""LiteLLM adapter — re-exports for convenient imports."""

from src.adapters.litellm.client import call_litellm, setup_litellm

__all__ = ["call_litellm", "setup_litellm"]
