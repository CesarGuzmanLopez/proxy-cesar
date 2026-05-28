# Image Delegation Bug Fix - Summary

## Problem Statement

Images sent to LLM models without vision capability were not being automatically described. Instead, they were passed through unchanged, causing models to respond with "No puedo ver imágenes" (I can't see images).

**Root Cause**: Content validation occurred BEFORE physical model selection, leading to incorrect decisions about whether content needed delegation.

## Solution Overview

Restructured the request pipeline to validate content AFTER selecting the specific physical model that will handle the request. This ensures that:
1. We check the CHOSEN model's capabilities, not just any model in the pseudo-model
2. If the chosen model lacks vision capability, images are auto-described via vision model
3. The description is sent to the LLM instead of raw image data

## Changes Made

### 1. `proxy/src/service/compatibility.py`

**Added**: New function `validate_physical_model_content()`

```python
def validate_physical_model_content(
    turn_caps: TurnCapabilities,
    physical_model,
) -> dict | None:
    """Validate if a SPECIFIC physical model can handle the incoming content.
    
    Used AFTER physical model selection to determine if content delegation
    (image description, audio transcription, etc.) is needed.
    
    Returns:
      - None if the physical model can handle all content
      - {"action": "transform_unsupported"} if content needs delegation
    """
```

This function checks if the selected physical model supports required capabilities:
- Images → checks `vision`
- Audio → checks `audio`
- PDF → checks `vision`
- Video → checks `video`

**Existing function preserved**: `validate_incoming_content()` remains for backward compatibility with other code paths.

### 2. `proxy/src/service/chat_service.py`

**Modified**: `_resolve_and_validate()` function (lines 372-391)
- Now returns only `(model_name, pseudo_model_schema, turn_capabilities)`
- Removed: Content validation logic (deferred to after model selection)
- Added import of `validate_physical_model_content`

**Modified**: `_resolve_session_conv_and_models()` function (lines 402-475)
- Now returns 8 items instead of 7
- Added: `selected_phys_model` (PhysicalModelSchema object) in return tuple
- Position: Item 5 (after `provider`, before `tools_filter`)

**Modified**: `process_chat_request()` function (lines 89-104, 121-161)
- Restructured pipeline order:
  1. Detect capabilities (no validation)
  2. Select physical model
  3. Validate physical model against capabilities
  4. Apply content delegation
  5. Continue with rest of pipeline

### 3. `proxy/src/api/chat_stream_persistence.py`

**Modified**: `_resolve_physical_model()` function (lines 37-59)
- Now returns 3 items instead of 2
- Added: `selected_phys` (PhysicalModelSchema object) in return tuple
- Updated return type annotation: `tuple[str, str | None, object]`

### 4. `proxy/src/api/chat_streaming.py`

**Modified**: Streaming path (lines 152-243)
- Removed: Early content validation (lines 155-165)
- Added: Content validation after physical model selection (lines 223-230)
- Updated model resolution to unpack the new physical model object from `_resolve_physical_model()`

**Updated import**: Changed from `validate_incoming_content` to `validate_physical_model_content`

## Request Flow Comparison

### Before (Broken)
```
User sends image to pseudo-model "normal"
  ↓
_resolve_and_validate()
  → Check: does ANY model in "normal" have vision? YES (Groq has vision)
  → Decision: No delegation needed
  ↓
_apply_content_delegation()
  → Returns messages unchanged
  ↓
_resolve_session_conv_and_models()
  → Selects Kimi-k2.5 (no vision!) as physical model
  ↓
LLM receives image → Error: "No puedo ver imágenes"
```

### After (Fixed)
```
User sends image to pseudo-model "normal"
  ↓
_resolve_and_validate()
  → Detect: message has images
  → NO VALIDATION YET (deferred)
  ↓
_resolve_session_conv_and_models()
  → Selects Kimi-k2.5 as physical model
  ↓
validate_physical_model_content()
  → Check: does Kimi-k2.5 have vision? NO
  → Decision: Image needs delegation
  ↓
_apply_content_delegation()
  → Describes image via Groq vision model
  → Stores in Valkey as [BLOB:hash]
  ↓
LLM receives: "Image described as: [description text]" → Works!
```

## Testing

### Unit Tests Verified
✓ Image to non-vision model → delegation triggered
✓ Image to vision model → no delegation needed
✓ Audio to non-audio model → delegation triggered
✓ Mixed capabilities handled correctly

### Integration Testing Needed
- [ ] E2E test: image to "fast" pseudo-model (defaults to Kimi-k2.5)
- [ ] E2E test: streaming + image delegation
- [ ] E2E test: multiple images in single request
- [ ] Regression: existing conversations (pseudo-model switch) still work
- [ ] Regression: text-only requests unaffected

## Backward Compatibility

✓ **No breaking changes to API**
- Request format unchanged
- Response format unchanged
- `proxy_metadata` fields unchanged

✓ **Existing code paths preserved**
- `validate_incoming_content()` still available for other uses
- All other functions maintain same external interface
- Only internal ordering changed

## Logging & Observability

The fix works with existing logging infrastructure:
- `content_delegation_applied` logs show which images were described
- `proxy_metadata.images_described` shows description count
- `proxy_metadata.images_described_by` shows which vision model was used

## Files Modified

1. `proxy/src/service/compatibility.py` - NEW function
2. `proxy/src/service/chat_service.py` - Restructured pipeline
3. `proxy/src/api/chat_stream_persistence.py` - Return physical model object
4. `proxy/src/api/chat_streaming.py` - Updated streaming path

Total changes: ~100 lines added, ~30 lines removed/refactored.
