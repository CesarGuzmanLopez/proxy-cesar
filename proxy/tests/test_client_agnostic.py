"""Client-agnostic model name normalization tests (Feature).

Verifies that ANY client (OpenCode, Cline, RooCode, Continue, Aider, curl)
can send model names in various formats and the proxy normalizes them correctly.
"""

import pytest

from src.config.pseudo_models import load_config
from src.service.model_resolver import normalize_model_name

# Load config once
_CONFIG = load_config()


@pytest.mark.parametrize(
    "raw_name,expected",
    [
        # Direct pseudo-model names
        ("normal", "normal"),
        ("tareas-avanzadas", "tareas-avanzadas"),
        ("pensamiento-profundo-caro", "pensamiento-profundo-caro"),
        ("flash", "flash"),
        ("vision", "vision"),
        ("compactador", "compactador"),
        # OpenCode local provider: "local/<pseudo>"
        ("local/normal", "normal"),
        # Custom provider names: "cesar-proxy/<pseudo>"
        ("cesar-proxy/normal", "normal"),
        ("cesar-proxy/tareas-avanzadas", "tareas-avanzadas"),
        # Cline fork: "cline/<pseudo>"
        ("cline/normal", "normal"),
        ("cline/flash", "flash"),
        # RooCode fork: "roo/<pseudo>"
        ("roo/normal", "normal"),
        ("roo/flash", "flash"),
        # OpenAI aliases (with and without prefix)
        ("gpt-4o", "normal"),
        ("local/gpt-4o", "normal"),
        ("gpt-4o-mini", "normal"),
        ("local/gpt-4o-mini", "normal"),
        ("o3", "pensamiento-profundo-caro"),
        ("local/o3", "pensamiento-profundo-caro"),
        ("o4-mini", "normal"),
        # Google alias
        ("gemini-2.5-flash", "vision"),
        ("local/gemini-2.5-flash", "vision"),
        # Default fallback for unknown models → 'flash'
        ("unknown-model", "flash"),
        ("local/unknown-thing", "local/unknown-thing"),
    ],
)
def test_model_name_normalization(raw_name, expected):
    """All model name formats should resolve to the correct pseudo-model."""
    result = normalize_model_name(raw_name, _CONFIG)
    assert result == expected, (
        f"'{raw_name}' should resolve to '{expected}', got '{result}'"
    )


def test_unknown_model_with_prefix_passthrough():
    """Unknown model with prefix is used as-is (passthrough)."""
    result = normalize_model_name("some-client/unknown", _CONFIG)
    assert result == "some-client/unknown"  # passthrough directo


def test_all_known_pseudo_models_normalize_to_themselves():
    """Every pseudo-model name normalizes to itself."""
    for name in _CONFIG.pseudo_models:
        assert normalize_model_name(name, _CONFIG) == name


def test_aliases_map_to_valid_pseudo_models():
    """All aliases must map to pseudo-models that exist in config."""
    for alias, target in _CONFIG.model_aliases.items():
        if alias == "default":
            continue
        assert target in _CONFIG.pseudo_models, (
            f"Alias '{alias}' maps to '{target}' which is not a valid pseudo-model"
        )


def test_default_alias_maps_to_valid_pseudo_model():
    """The default alias must map to a valid pseudo-model."""
    assert "default" in _CONFIG.model_aliases
    assert _CONFIG.model_aliases["default"] in _CONFIG.pseudo_models
