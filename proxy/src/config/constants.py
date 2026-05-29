"""Configurable constants for image processing, blob storage, and performance tuning.

These values are extracted from hardcoded magic numbers for easier configuration
and performance tuning without code changes.
"""

# Image Processing Constants
# Maximum image dimension (width or height) before resize
MAX_IMAGE_DIMENSION = 840

# JPEG compression quality (0-100) after image degradation
# 85 provides good balance between size and quality
IMAGE_JPEG_QUALITY = 85

# Tokens per vision tile in OpenAI vision model cost calculation
# Reference: https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding
VISION_TOKENS_PER_TILE = 170

# Fixed tokens for low-detail vision images in OpenAI
VISION_LOW_DETAIL_TOKENS = 85

# LLM Call Constants
# Default timeout for LiteLLM calls (in seconds)
# Protects against hanging requests through the fallback chain
DEFAULT_LLM_TIMEOUT_SECONDS = 60
# Blob Storage Constants
# TTL for base64-encoded blobs in Redis (in seconds)
# 86400 = 24 hours — allows multi-turn conversations within a day
BLOB_STORAGE_TTL_SECONDS = 86400

# Compaction Job Constants
# Hard timeout for compaction jobs (in seconds)
# 300 = 5 minutes — increase if large conversations timeout
COMPACTION_JOB_TIMEOUT_SECONDS = 300
