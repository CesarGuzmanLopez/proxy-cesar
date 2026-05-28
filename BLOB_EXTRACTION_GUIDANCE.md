# Blob Extraction Guidance - Model Tool Support

## Overview

When the proxy auto-extracts content (describes images, transcribes audio, extracts PDF text), it now explicitly tells the model that the content was processed and that the model can use its own specialized tools if available.

## What Changed

### 1. Enhanced Blob Metadata Messages

**Before:**
```
[The user sent an image. blob: BLOB:abc123:image/jpeg | 45 KB
Image of a cat sitting on a desk
]
```

**After:**
```
[Content provided: image
  blob_ref: BLOB:abc123:image/jpeg | size: 45 KB
  extraction: Vision model (description via Groq/similar)

Extracted content:
A cat sitting on a wooden desk, looking at the camera

Note: If you have specialized tools for analyzing this content, you can:
  • Use the blob reference to retrieve raw data if needed
  • Apply your own analysis tools for custom extraction
  • Combine your analysis with the provided extraction
]
```

### 2. System Message with Tool Guidance

When blobs are detected, a system message is automatically injected:

```
**Blob Content Processing Guide**

Some of the content in this conversation was auto-extracted from multimodal files
(images, audio, PDFs) that the current model cannot process natively.
The proxy has automatically:
  • Described images using vision models
  • Transcribed audio using speech-to-text
  • Extracted text from PDFs

**If you have specialized tools available**, you can:
  1. Use the blob reference (format: BLOB:hash:mimetype) to access raw data
  2. Apply your own analysis for custom extraction or more detailed processing
  3. Combine your specialized analysis with the provided extraction

The extracted content is provided as text context above.
Use your tools if you need alternative analysis or higher precision.
```

## How It Works

### Scenario 1: Model WITHOUT Specialized Tools

User sends image to Kimi (text-only model):
```
Flow:
  Proxy: Detects image, validates Kimi has no vision
    ↓
  Proxy: Auto-describes image via Groq Vision
    ↓
  Proxy: Sends blob metadata + description to Kimi
    ↓
  Proxy: Injects system message explaining extraction
    ↓
  Kimi: Receives description and system guidance
    → "I understand the image was auto-described"
    → "I don't have image processing tools"
    → Uses the provided description for analysis
```

### Scenario 2: Model WITH Specialized Tools

User sends PDF to Claude (has specialized PDF tool):
```
Flow:
  Proxy: Detects PDF, validates Claude has no vision
    ↓
  Proxy: Auto-extracts PDF text using PDF extractor
    ↓
  Proxy: Sends blob reference + extracted text to Claude
    ↓
  Proxy: Injects system message explaining extraction
    ↓
  Claude: Receives extraction and system guidance
    → "I understand the PDF was auto-extracted"
    → "I have specialized PDF analysis tools"
    → Can EITHER:
       a) Use the provided text extraction
       b) Use its own tools to analyze the blob reference for custom parsing
       c) Combine both approaches
```

### Scenario 3: Vision Model (No Extraction Needed)

User sends image to Groq (vision=true):
```
Flow:
  Proxy: Detects image, validates Groq has vision
    ↓
  Proxy: No extraction needed
    ↓
  Proxy: Sends image directly to Groq
    ↓
  Proxy: No system message injected (no blobs)
    ↓
  Groq: Receives image natively
    → Processes directly without extraction
```

## Technical Implementation

### File Changes

1. **`src/service/tool_detector.py`**
   - Enhanced `_build_blob_output()` to include:
     - Blob reference (hash, MIME type, size)
     - Extraction method used (Vision, Whisper, PDF extractor)
     - Extracted content
     - Guidance on using specialized tools
   
   - Added `inject_blob_extraction_guidance()` function:
     - Detects if messages contain blobs
     - Creates system message with tool guidance
     - Only injects if no system message already exists

2. **`src/service/chat_service.py`**
   - Calls `inject_blob_extraction_guidance()` after `_apply_content_delegation()`
   - Ensures all blob messages have proper guidance

3. **`src/api/chat_streaming.py`**
   - Calls `inject_blob_extraction_guidance()` for streaming path
   - Ensures consistency between streaming and non-streaming

## Benefits

### For Users with Specialized Tools
- Models can choose to use proxy extraction OR their own tools
- More control over content analysis
- Better accuracy for domain-specific needs

### For Users without Tools
- No impact - models use proxy extraction as before
- Extra context helps understand content processing

### For Proxy Transparency
- Explicit about what was auto-extracted
- Models know this wasn't user-provided
- Clear path to using alternatives

## Usage Examples

### Example 1: Complex PDF Analysis

```
User: "Analyze this financial report"
  ↓ [User sends PDF to Kimi]
  ↓
Proxy: Auto-extracts PDF → "Company Revenue: $5M, Expenses: $3M, Profit: $2M"
  ↓
System Message: Tells Kimi it can use custom PDF tools if available
  ↓
Kimi: Uses provided extraction for analysis (text model, no PDF tools)
  → "Based on the extracted report, profit margin is 40%..."
```

### Example 2: Detailed Image Analysis

```
User: "Identify objects and their relationships in this image"
  ↓ [User sends complex image to Claude with vision tool]
  ↓
Proxy: Auto-describes image via Groq Vision
  → "A person holding a coffee cup while looking at a laptop screen"
  ↓
System Message: Tells Claude it can use its vision tools if available
  ↓
Claude: Recognizes it has specialized vision tools
  → Uses blob reference to access raw image
  → Runs detailed object detection
  → Identifies person, cup (ceramic), laptop (Apple), relationships (holding)
  → Provides detailed analysis beyond the extraction
```

### Example 3: Multi-Modal Content

```
User: "Analyze this presentation: [image slide] + [PDF notes]"
  ↓ [User sends mixed content to Qwen]
  ↓
Proxy: 
  - Extracts image: "Slide showing sales graph with 3-month trend"
  - Extracts PDF: "Q3 Notes: Sales increased 15% YoY"
  ↓
System Message: Explains both extractions + tool availability
  ↓
Qwen: 
  → Receives both descriptions
  → Uses text extraction for analysis
  → Output: "Sales growth of 15% YoY as shown in slide and confirmed by notes"
```

## Edge Cases Handled

1. **No Blobs**: No system message injected (clean conversation)
2. **Existing System Message**: New message NOT injected (respects user's custom instructions)
3. **Mixed Content**: System message explains ALL types (images, audio, PDFs)
4. **Streaming Path**: Same guidance as non-streaming
5. **Multiple Blobs**: Single system message covers all (no repetition)

## Configuration

The system message:
- ✓ Only injected when blobs detected
- ✓ Only injected once (respects existing system messages)
- ✓ Works with any pseudo-model
- ✓ Works with any content type
- ✓ Language is neutral (models from different origins understand it)

## Future Enhancements

1. **Configurable Message**: Allow customizing guidance text per model
2. **Tool Registry**: Know which models have which tools
3. **Smart Routing**: Route directly to vision models if available
4. **Cost Optimization**: Avoid extraction if model has native capability
5. **Quality Assessment**: Track when tool extraction is preferred over proxy extraction

## Testing

```bash
# Test that system message is injected when blobs present
pytest tests/test_blob_extraction_guidance.py -xvs -k "system_message"

# Test that no message injected for vision models (no extraction)
pytest tests/test_blob_extraction_guidance.py -xvs -k "no_extraction"

# Test that existing system messages are respected
pytest tests/test_blob_extraction_guidance.py -xvs -k "existing_system"
```

## Summary

The proxy now explicitly communicates to models:
1. **What was extracted**: Images described, audio transcribed, PDFs extracted
2. **How it was extracted**: Vision model, Whisper, PDF extractor
3. **Available alternatives**: Models can use their own tools if available
4. **How to access raw data**: Blob reference format for direct access

This enables intelligent delegation where:
- Simple models use proxy extraction
- Advanced models use their specialized tools
- All models understand the processing pipeline
