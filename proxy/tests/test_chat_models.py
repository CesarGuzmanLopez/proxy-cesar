"""Tests for chat models (StreamContext, SaveContext, MetadataContext).

Covers default values for Sprint 5 fields.
"""

from src.service.chat_models import (
    FallbackInfo,
    MetadataContext,
    SaveContext,
    StreamContext,
)
from src.domain.capabilities import SessionCapabilities


def test_stream_context_defaults():
    """StreamContext default values after construction with required args."""
    ctx = StreamContext(
        litellm_response="mock",
        conversation_id="test-id",
        pseudo_model="test",
        physical_model="test-phys",
    )
    assert ctx.fallback_info is None
    assert ctx.affinity_maintained is True
    assert ctx.context_window is None
    assert ctx.session_caps is None
    assert ctx.compatibility_warning is None
    assert ctx.compatibility_details is None
    assert ctx.tools_filter_applied is False
    assert ctx.tools_filter_reason is None
    assert ctx.images_described == 0
    assert ctx.images_described_by is None
    assert ctx.router_suggestion is None
    assert ctx.db is None
    assert ctx.conv is None
    assert ctx.conv_uuid is None
    assert ctx.turn_caps is None
    assert ctx.provider is None
    assert ctx.messages is None
    assert ctx.tools is None
    assert ctx.tool_choice is None
    assert ctx.resolved_model is None
    assert ctx.is_new is None


def test_save_context_defaults():
    """SaveContext default values for Sprint 5 fields."""
    caps = SessionCapabilities(conversation_id="test")
    ctx = SaveContext(
        db="mock",
        conv="mock",
        conv_uuid="mock",
        conv_id="test",
        pseudo_model_name="test",
        physical_model="test",
        provider="test",
        turn_caps=caps,
        messages=[],
        response="mock",
        fallback_info=FallbackInfo(),
        is_new_conversation=True,
        existing_affinity=None,
        pm_schema="mock",
        session_caps=caps,
        tools=None,
        tool_choice=None,
        compatibility={},
        tools_filter={},
    )
    assert ctx.images_described == 0
    assert ctx.images_described_by is None
    assert ctx.router_suggestion is None


def test_metadata_context_defaults():
    """MetadataContext default values."""
    ctx = MetadataContext(
        pseudo_model="test",
        physical_model="test-phys",
        conversation_id="test-id",
    )
    assert ctx.context_tokens == 0
    assert ctx.context_window is None
    assert ctx.fallback_info is None
    assert ctx.affinity_maintained is True
    assert ctx.session_caps is None
    assert ctx.compatibility_warning is None
    assert ctx.compatibility_details is None
    assert ctx.tools_filter_applied is False
    assert ctx.tools_filter_reason is None
    assert ctx.images_described == 0
    assert ctx.images_described_by is None
    assert ctx.images_degraded_manually is False
    assert ctx.router_suggestion is None
