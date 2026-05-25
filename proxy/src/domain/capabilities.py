"""Domain types for capability detection and compatibility validation.

Pure dataclasses with no infrastructure dependencies.
python.md §1.1: domain must not import FastAPI, SQLAlchemy, or Pydantic.
"""

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class TurnCapabilities:
    """Capabilities detected in a single turn's messages.

    These flags are computed from the raw messages array.
    Rules are deterministic — no ML, no heuristics, no fuzzy matching.

    Sprint 3 extensions: tools_incomplete, thinking_blocks, tools_level_used.
    """

    has_images: bool = False
    has_audio: bool = False
    has_pdf: bool = False
    has_video: bool = False
    has_tools: bool = False
    has_parallel_tools: bool = False

    # Sprint 3: tool-specific state
    tools_incomplete: bool = False
    thinking_blocks: dict | None = None
    tools_level_used: int = (
        0  # ToolLevel as int (0=NONE, 1=BASIC, 2=STANDARD, 3=PARALLEL_STRICT)
    )

    # Sprint 5: image description tracking per turn
    images_described_count: int = 0
    images_described_by: str | None = None
    images_degraded_manually: bool = False


@dataclass
class SessionCapabilities:
    """Accumulated capabilities across all turns of a conversation.

    Flags are additive — once set to True they are NEVER reset.
    (except via explicit operations like normalize-tools or degrade-images)

    Sprint 3 extension: max_tools_level tracks highest tool complexity used.
    """

    conversation_id: str
    has_images: bool = False
    has_audio: bool = False
    has_pdf: bool = False
    has_video: bool = False
    has_tools: bool = False
    has_parallel_tools: bool = False
    total_tokens: int = 0

    # Sprint 3: highest tool level ever used in this conversation
    max_tools_level: int = 0

    # Sprint 5: image degradation tracking (cumulative, never reset)
    images_described: int = 0
    images_degraded_manually: bool = False

    def merge(self, turn_caps: TurnCapabilities) -> "SessionCapabilities":
        """Merge new turn capabilities into session (additive only)."""
        self.has_images = self.has_images or turn_caps.has_images
        self.has_audio = self.has_audio or turn_caps.has_audio
        self.has_pdf = self.has_pdf or turn_caps.has_pdf
        self.has_video = self.has_video or turn_caps.has_video
        self.has_tools = self.has_tools or turn_caps.has_tools
        self.has_parallel_tools = (
            self.has_parallel_tools or turn_caps.has_parallel_tools
        )
        self.max_tools_level = max(self.max_tools_level, turn_caps.tools_level_used)
        # Sprint 5: images_described is additive (count of described images)
        self.images_described += turn_caps.images_described_count
        self.images_degraded_manually = (
            self.images_degraded_manually or turn_caps.images_degraded_manually
        )
        return self


class CompatibilityStatus(str, Enum):
    """Result of a pseudo-model switch compatibility check."""

    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class CompatibilityResult:
    """Result of validating a pseudo-model switch.

    Attributes:
        status: SAFE, WARNING, or BLOCKED
        reason: Human-readable explanation
        remediation: List of actionable options for the user (BLOCKED only)
        details: Extra structured data (e.g., which models are affected)
    """

    status: CompatibilityStatus
    reason: str
    remediation: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    # Sprint 5: carries auto-described messages from validate_switch → caller
    # Avoids re-scanning the full conversation history for images.
    auto_described_messages: list[dict] | None = None
    auto_describe_metadata: dict | None = None
