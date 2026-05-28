"""Test image delegation fix - validates images are auto-described for non-vision models.

This test verifies the fix for the bug where images sent to non-vision models
were not being auto-described, causing "No puedo ver imágenes" errors.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.service.compatibility import validate_physical_model_content
from src.domain.capabilities import TurnCapabilities


class TestImageDelegationValidation:
    """Unit tests for the new validate_physical_model_content function."""

    def test_image_to_non_vision_model(self):
        """Images to non-vision models should trigger delegation."""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)

        class NonVisionModel:
            vision = False
            audio = False
            video = False

        result = validate_physical_model_content(turn_caps, NonVisionModel())
        assert result == {"action": "transform_unsupported"}

    def test_image_to_vision_model(self):
        """Images to vision models should NOT trigger delegation."""
        turn_caps = TurnCapabilities(conversation_id="test", has_images=True)

        class VisionModel:
            vision = True
            audio = False
            video = False

        result = validate_physical_model_content(turn_caps, VisionModel())
        assert result is None

    def test_audio_to_non_audio_model(self):
        """Audio to non-audio models should trigger delegation."""
        turn_caps = TurnCapabilities(conversation_id="test", has_audio=True)

        class NonAudioModel:
            vision = False
            audio = False
            video = False

        result = validate_physical_model_content(turn_caps, NonAudioModel())
        assert result == {"action": "transform_unsupported"}

    def test_audio_to_audio_model(self):
        """Audio to audio models should NOT trigger delegation."""
        turn_caps = TurnCapabilities(conversation_id="test", has_audio=True)

        class AudioModel:
            vision = False
            audio = True
            video = False

        result = validate_physical_model_content(turn_caps, AudioModel())
        assert result is None

    def test_pdf_to_non_vision_model(self):
        """PDFs to non-vision models should trigger delegation."""
        turn_caps = TurnCapabilities(conversation_id="test", has_pdf=True)

        class NonVisionModel:
            vision = False
            audio = False
            video = False

        result = validate_physical_model_content(turn_caps, NonVisionModel())
        assert result == {"action": "transform_unsupported"}

    def test_pdf_to_vision_model(self):
        """PDFs to vision models should NOT trigger delegation."""
        turn_caps = TurnCapabilities(conversation_id="test", has_pdf=True)

        class VisionModel:
            vision = True
            audio = False
            video = False

        result = validate_physical_model_content(turn_caps, VisionModel())
        assert result is None

    def test_video_to_non_video_model(self):
        """Videos to non-video models should trigger delegation."""
        turn_caps = TurnCapabilities(conversation_id="test", has_video=True)

        class NonVideoModel:
            vision = False
            audio = False
            video = False

        result = validate_physical_model_content(turn_caps, NonVideoModel())
        assert result == {"action": "transform_unsupported"}

    def test_video_to_video_model(self):
        """Videos to video models should NOT trigger delegation."""
        turn_caps = TurnCapabilities(conversation_id="test", has_video=True)

        class VideoModel:
            vision = False
            audio = False
            video = True

        result = validate_physical_model_content(turn_caps, VideoModel())
        assert result is None

    def test_mixed_content_partial_support(self):
        """Model with partial support should trigger delegation for missing capability."""
        turn_caps = TurnCapabilities(
            conversation_id="test", has_images=True, has_audio=False
        )

        class AudioOnlyModel:
            vision = False
            audio = True
            video = False

        result = validate_physical_model_content(turn_caps, AudioOnlyModel())
        assert result == {"action": "transform_unsupported"}

    def test_text_only_no_delegation(self):
        """Text-only messages should never trigger delegation."""
        turn_caps = TurnCapabilities(
            conversation_id="test",
            has_images=False,
            has_audio=False,
            has_pdf=False,
            has_video=False,
        )

        class LimitedModel:
            vision = False
            audio = False
            video = False

        result = validate_physical_model_content(turn_caps, LimitedModel())
        assert result is None

    def test_full_capability_model(self):
        """Model with all capabilities should never trigger delegation."""
        turn_caps = TurnCapabilities(
            conversation_id="test",
            has_images=True,
            has_audio=True,
            has_pdf=True,
            has_video=True,
        )

        class FullCapabilityModel:
            vision = True
            audio = True
            video = True

        result = validate_physical_model_content(turn_caps, FullCapabilityModel())
        assert result is None
