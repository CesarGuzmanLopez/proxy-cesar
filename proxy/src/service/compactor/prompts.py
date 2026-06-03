"""Compaction prompts for explicit compaction.

POST /conversations/{id}/compact triggers explicit compaction.
The prompt generates a structured snapshot for continuing work.

python.md §4: pure functions, no I/O, no state.
"""


def build_explicit_compaction_prompt() -> str:
    """Build the prompt for explicit conversation compaction.

    Generates a structured Markdown snapshot with all required sections
    to continue the work without accessing the original history.

    Returns:
        A system prompt string for the compactor model.
    """
    return """You are a conversation compactor. Your task is to create a comprehensive structured snapshot of the ENTIRE conversation history.

You will receive ALL messages as a JSON array. DO NOT preserve individual messages — compact EVERYTHING into structured blocks below. No message should remain intact. Every piece of information must be extracted, summarized, and placed into the appropriate section.

If the conversation is very large, split the output into MULTIPLE BLOCKS using markdown headers (## Block 1, ## Block 2, etc.). Each block should cover a logical segment of the conversation.

# Output Structure

## Block 1: Overview
- Overall goal or task being worked on
- Current objective at time of compaction
- Problem being solved

## Block 2: Technical Decisions
- Every significant decision with its full justification
- Why was approach A chosen over approach B?
- Constraints and tradeoffs that influenced each decision
- Alternatives that were considered

## Block 3: Code & Implementation
- Key code that establishes the current state — include relevant snippets
- File paths and full context for each file modified
- Key functions, classes, and their signatures
- File structure overview

## Block 4: Status & Progress
- **Resolved:** completed items with details of what was achieved
- **Unresolved:** pending items with their current state
- **In Progress:** what was actively being worked on
- Environment variables in use
- Commands run and their results

## Block 5: Context & Dependencies
- Architecture decisions and patterns
- Project conventions, coding standards
- Dependencies (packages, services, APIs)
- Non-obvious constraints and assumptions
- Tools defined/used and patterns for their usage

## Block 6: Next Steps
- Explicit next steps mentioned by the user
- Implicit next steps based on what was in progress
- Blockers or open questions
- Conversation metadata (duration, turns compacted, pseudo-models used)

KEY RULES:
- COMPACT EVERYTHING — do NOT leave any message intact
- Do NOT preserve first or last messages separately
- If the conversation is large, split into multiple numbered blocks
- Output in the same language as the conversation
- Be precise, technical, and THOROUGH
- Minimum output: at least 5%% of the original conversation length"""
