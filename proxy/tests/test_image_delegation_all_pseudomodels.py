"""Integration test: image delegation for all pseudo-models.

Verifies that image delegation works correctly for every pseudo-model,
regardless of whether the physical model has vision capability.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from src.service.compatibility import validate_physical_model_content
from src.domain.capabilities import TurnCapabilities


# Mock physical models from pseudo_models.yaml
class KimiK25:
    """Kimi K2.5 - no vision"""
    vision = False
    audio = False
    video = False
    model = "openai/kimi-k2.5"


class QwenMax:
    """Qwen 3.7 Max - no vision"""
    vision = False
    audio = False
    video = False
    model = "anthropic/qwen3.7-max"


class DeepSeekV4:
    """DeepSeek V4 - no vision"""
    vision = False
    audio = False
    video = False
    model = "deepseek-v4-pro"


class GroqVision:
    """Groq with vision - HAS vision"""
    vision = True
    audio = False
    video = False
    model = "groq/llama-4-scout-17b-16e-instruct"


class Claude:
    """Claude - HAS vision"""
    vision = True
    audio = False
    video = False
    model = "anthropic/claude-opus-4-1"


class O3Mini:
    """O3 Mini - no vision"""
    vision = False
    audio = False
    video = False
    model = "openai/o3-mini"


class TestImageDelegationAllPseudomodels:
    """Test image delegation for all pseudo-models in pseudo_models.yaml"""

    # ── Pensamiento Profundo (vision=false) ──────────────────────────────────

    def test_pensamiento_profundo_caro_qwen(self):
        """Pensamiento Profundo - Qwen 3.7 Max (no vision) should delegate"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)
        model = QwenMax()
        result = validate_physical_model_content(turn_caps, model)
        assert result == {"action": "transform_unsupported"}, \
            "Pensamiento Profundo with Qwen should delegate images"

    def test_pensamiento_profundo_caro_deepseek(self):
        """Pensamiento Profundo - DeepSeek V4 Pro (no vision) should delegate"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)
        model = DeepSeekV4()
        result = validate_physical_model_content(turn_caps, model)
        assert result == {"action": "transform_unsupported"}, \
            "Pensamiento Profundo with DeepSeek should delegate images"

    # ── Pensamiento Rápido (vision=false) ──────────────────────────────────

    def test_pensamiento_rapido(self):
        """Pensamiento Rápido - should delegate images for all models"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)

        # Qwen 3.6 Plus (no vision)
        model_qwen = Mock(vision=False, audio=False, video=False)
        result = validate_physical_model_content(turn_caps, model_qwen)
        assert result == {"action": "transform_unsupported"}

        # DeepSeek V4 Flash (no vision)
        model_deepseek = Mock(vision=False, audio=False, video=False)
        result = validate_physical_model_content(turn_caps, model_deepseek)
        assert result == {"action": "transform_unsupported"}

    # ── Tareas Avanzadas (vision=false) ──────────────────────────────────

    def test_tareas_avanzadas(self):
        """Tareas Avanzadas - Kimi K2.6 (no vision) should delegate"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)
        model = Mock(vision=False, audio=False, video=False)
        result = validate_physical_model_content(turn_caps, model)
        assert result == {"action": "transform_unsupported"}

    # ── Código Rápido (vision=false) ──────────────────────────────────

    def test_codigo_rapido(self):
        """Código Rápido - Kimi K2.5 (no vision) should delegate images"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)
        model = KimiK25()
        result = validate_physical_model_content(turn_caps, model)
        assert result == {"action": "transform_unsupported"}, \
            "Código Rápido with Kimi K2.5 should delegate images"

    # ── Código Avanzado (vision=false) ──────────────────────────────────

    def test_codigo_avanzado(self):
        """Código Avanzado - Kimi K2.6 (no vision) should delegate"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)
        model = Mock(vision=False, audio=False, video=False)
        result = validate_physical_model_content(turn_caps, model)
        assert result == {"action": "transform_unsupported"}

    # ── Razonador Caro (vision=false) ──────────────────────────────────

    def test_razonador_caro(self):
        """Razonador Caro - O3 Mini (no vision) should delegate"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)
        model = O3Mini()
        result = validate_physical_model_content(turn_caps, model)
        assert result == {"action": "transform_unsupported"}

    # ── Vision (vision=true - should NOT delegate) ──────────────────────────

    def test_vision_groq(self):
        """Vision - Groq (vision=true) should NOT delegate"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)
        model = GroqVision()
        result = validate_physical_model_content(turn_caps, model)
        assert result is None, \
            "Vision pseudo-model with Groq should NOT delegate (has native vision)"

    # ── Multimodal (vision=true) ──────────────────────────────────────

    def test_multimodal_claude(self):
        """Multimodal - Claude (vision=true) should NOT delegate"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)
        model = Claude()
        result = validate_physical_model_content(turn_caps, model)
        assert result is None, \
            "Multimodal with Claude should NOT delegate (has native vision)"

    # ── Audio Tests ─────────────────────────────────────────────────────────

    def test_audio_to_non_audio_models(self):
        """All non-audio models should delegate audio"""
        turn_caps = TurnCapabilities(conversation_id="test", has_audio=True)

        # Most models don't have audio support
        models = [
            KimiK25(), QwenMax(), DeepSeekV4(), O3Mini(),
            Mock(vision=False, audio=False, video=False),
        ]

        for model in models:
            result = validate_physical_model_content(turn_caps, model)
            assert result == {"action": "transform_unsupported"}, \
                f"{model} should delegate audio"

    # ── PDF Tests ───────────────────────────────────────────────────────────

    def test_pdf_to_non_vision_models(self):
        """PDFs to non-vision models should delegate (need text extraction)"""
        turn_caps = TurnCapabilities(conversation_id="test", has_pdf=True)

        non_vision_models = [
            KimiK25(), QwenMax(), DeepSeekV4(), O3Mini(),
        ]

        for model in non_vision_models:
            result = validate_physical_model_content(turn_caps, model)
            assert result == {"action": "transform_unsupported"}, \
                f"{model.model} should delegate PDFs"

    def test_pdf_to_vision_models(self):
        """PDFs to vision models should NOT delegate"""
        turn_caps = TurnCapabilities(conversation_id="test", has_pdf=True)

        vision_models = [GroqVision(), Claude()]

        for model in vision_models:
            result = validate_physical_model_content(turn_caps, model)
            assert result is None, \
                f"{model.model} should NOT delegate PDFs (can read them)"

    # ── Mixed Content Tests ─────────────────────────────────────────────────

    def test_mixed_image_and_audio_no_vision_no_audio(self):
        """Mixed image+audio to models without either should delegate"""
        turn_caps = TurnCapabilities(
            conversation_id="test",
            has_images=True,
            has_audio=True
        )
        model = KimiK25()
        result = validate_physical_model_content(turn_caps, model)
        assert result == {"action": "transform_unsupported"}

    def test_mixed_image_and_audio_has_vision_no_audio(self):
        """Mixed image+audio to vision-only model should delegate (for audio)"""
        turn_caps = TurnCapabilities(
            conversation_id="test",
            has_images=True,
            has_audio=True
        )
        model = GroqVision()  # has vision but not audio
        result = validate_physical_model_content(turn_caps, model)
        assert result == {"action": "transform_unsupported"}, \
            "Should delegate audio even if image support exists"

    # ── Text-only (no delegation needed) ────────────────────────────────────

    def test_text_only_no_model_can_handle_it(self):
        """Text-only should never trigger delegation"""
        turn_caps = TurnCapabilities(
            conversation_id="test",
            has_images=False,
            has_audio=False,
            has_pdf=False,
            has_video=False,
        )

        # Even models with no capabilities should not trigger delegation for text
        model = Mock(vision=False, audio=False, video=False)
        result = validate_physical_model_content(turn_caps, model)
        assert result is None, "Text-only should not trigger delegation"

    # ── Full capability model ────────────────────────────────────────────────

    def test_full_capability_model(self):
        """Model with all capabilities should never delegate"""
        turn_caps = TurnCapabilities(
            conversation_id="test",
            has_images=True,
            has_audio=True,
            has_pdf=True,
            has_video=True,
        )
        model = Mock(vision=True, audio=True, video=True)
        result = validate_physical_model_content(turn_caps, model)
        assert result is None, "Full capability model should not delegate"

    # ── Edge cases ──────────────────────────────────────────────────────────

    def test_model_with_missing_attributes(self):
        """Model with missing capability attributes should be treated as not having them"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)

        # Mock model with no vision attribute (should be treated as False)
        model = Mock(spec=[])  # No attributes at all
        result = validate_physical_model_content(turn_caps, model)
        assert result == {"action": "transform_unsupported"}

    def test_none_capability_values(self):
        """None values for capabilities should be treated as False"""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)
        model = Mock(vision=None, audio=None, video=None)
        result = validate_physical_model_content(turn_caps, model)
        assert result == {"action": "transform_unsupported"}
