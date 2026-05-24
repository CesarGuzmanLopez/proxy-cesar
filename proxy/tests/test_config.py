"""Tests for pseudo-models YAML validation.

sprint §13.1 — minimum 15 test cases for the 14 validation rules.
"""

import sys
from pathlib import Path

import pytest
import yaml

from src.config.pseudo_models import (
    ProxyConfigSchema,
    load_config,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "pseudo_models.yaml"


def test_1_valid_config_loads():
    """Valid pseudo_models.yaml loads without error."""
    config = load_config(CONFIG_PATH)
    assert len(config.pseudo_models) == 8


def test_2_missing_file():
    """Missing file → SystemExit(1)."""
    with pytest.raises(SystemExit) as exc:
        load_config(Path("/nonexistent/pseudo_models.yaml"))
    assert exc.value.code == 1


def test_3_invalid_yaml(tmp_path):
    """Invalid YAML syntax → SystemExit(1)."""
    f = tmp_path / "bad.yaml"
    f.write_text("{invalid: yaml: broken: [}")
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_4_missing_pseudo_models_key(tmp_path):
    """Missing 'pseudo_models' key → SystemExit(1)."""
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump({"not_pseudo_models": {}}))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_5_empty_physical_models(tmp_path):
    """Empty physical_models list → validation error."""
    data = {
        "pseudo_models": {
            "test-model": {
                "display_name": "Test",
                "description": "test",
                "input_token_threshold": None,
                "context_window": None,
                "physical_models": [],
            }
        }
    }
    f = tmp_path / "empty.yaml"
    f.write_text(yaml.dump(data))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_6_missing_model_field(tmp_path):
    """Missing 'model' in physical model → validation error."""
    data = {
        "pseudo_models": {
            "test-model": {
                "display_name": "Test",
                "description": "test",
                "input_token_threshold": None,
                "context_window": None,
                "physical_models": [{"provider": "test"}],
            }
        }
    }
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump(data))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_7_openai_tools_compatible_false(tmp_path):
    """openai_tools_compatible: false → validation error."""
    data = {
        "pseudo_models": {
            "test-model": {
                "display_name": "Test",
                "description": "test",
                "input_token_threshold": None,
                "context_window": None,
                "physical_models": [
                    {
                        "provider": "test",
                        "model": "test-model",
                        "openai_tools_compatible": False,
                    }
                ],
            }
        }
    }
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump(data))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_8_invalid_fallback_strategy(tmp_path):
    """Invalid fallback_strategy → validation error."""
    data = {
        "pseudo_models": {
            "test-model": {
                "display_name": "Test",
                "description": "test",
                "input_token_threshold": None,
                "context_window": None,
                "physical_models": [{"provider": "test", "model": "m"}],
                "fallback_strategy": "invalid",
            }
        }
    }
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump(data))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_9_invalid_on_downgrade(tmp_path):
    """Invalid image_handling.on_downgrade → validation error."""
    data = {
        "pseudo_models": {
            "test-model": {
                "display_name": "Test",
                "description": "test",
                "input_token_threshold": None,
                "context_window": None,
                "image_handling": {"on_downgrade": "invalid"},
                "physical_models": [{"provider": "test", "model": "m"}],
            }
        }
    }
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump(data))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_10_continuous_compaction_no_trigger(tmp_path):
    """continuous_compaction enabled but missing trigger_pct → error."""
    data = {
        "pseudo_models": {
            "test-model": {
                "display_name": "Test",
                "description": "test",
                "input_token_threshold": None,
                "context_window": 100000,
                "continuous_compaction": {"enabled": True},
                "physical_models": [{"provider": "test", "model": "m"}],
            }
        }
    }
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump(data))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_11_pre_compaction_no_threshold(tmp_path):
    """pre_compaction enabled but missing threshold → error."""
    data = {
        "pseudo_models": {
            "test-model": {
                "display_name": "Test",
                "description": "test",
                "input_token_threshold": None,
                "context_window": 100000,
                "pre_compaction": {"enabled": True},
                "physical_models": [{"provider": "test", "model": "m"}],
            }
        }
    }
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump(data))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_12_pre_compaction_bad_compactor(tmp_path):
    """pre_compaction.compactor references unknown pseudo-model."""
    data = {
        "pseudo_models": {
            "test-model": {
                "display_name": "Test",
                "description": "test",
                "input_token_threshold": None,
                "context_window": 100000,
                "pre_compaction": {
                    "enabled": True,
                    "threshold": 1000,
                    "compactor": "nonexistent",
                },
                "physical_models": [{"provider": "test", "model": "m"}],
            }
        }
    }
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump(data))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_13_router_llm_bad_suggester(tmp_path):
    """router_llm.suggester references unknown pseudo-model."""
    data = {
        "pseudo_models": {
            "test-model": {
                "display_name": "Test",
                "description": "test",
                "input_token_threshold": None,
                "context_window": 100000,
                "router_llm": {
                    "enabled": True,
                    "suggester": "nonexistent",
                },
                "physical_models": [{"provider": "test", "model": "m"}],
            }
        }
    }
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump(data))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_14_extra_field_forbidden(tmp_path):
    """Extra unknown field in pseudo-model → extra='forbid' catches it."""
    data = {
        "pseudo_models": {
            "test-model": {
                "display_name": "Test",
                "description": "test",
                "input_token_threshold": None,
                "context_window": None,
                "physical_models": [{"provider": "test", "model": "m"}],
                "unknown_field": "should not be here",
            }
        }
    }
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump(data))
    with pytest.raises(SystemExit) as exc:
        load_config(f)
    assert exc.value.code == 1


def test_15_all_8_pseudo_models_loaded():
    """All 8 pseudo-models are loaded from the production YAML."""
    config = load_config(CONFIG_PATH)
    expected = [
        "pensamiento-profundo-caro",
        "tareas-avanzadas",
        "avanzada-vision",
        "normal",
        "deep-flash",
        "flash-lowcost",
        "flash-vision",
        "compactador",
    ]
    for name in expected:
        assert name in config.pseudo_models, f"Missing pseudo-model: {name}"
    assert len(config.pseudo_models) == 8
