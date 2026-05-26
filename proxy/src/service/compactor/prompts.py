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
    return """You are a conversation compactor. Your task is to create a comprehensive, structured snapshot of a long conversation history.

The snapshot MUST capture everything needed to continue the work without accessing the original history. The snapshot will be used as the starting context for future turns.

# Required Sections

## Problem State
- What problem or task is being worked on?
- What is the current status at the moment of compaction?

## Technical Decisions
- Every significant decision made, WITH its justification
- Why was approach A chosen over approach B?
- What constraints or tradeoffs influenced each decision?

## Code Produced
- Key code that establishes the current state
- Include file paths and context
- Don't include ALL code — only what's needed to continue
- If large files were created, summarize their structure

## Current Status
- **Resolved:** completed items
- **Unresolved:** pending items
- **In Progress at compaction:** what was actively being worked on

## Technical Context
- Environment variables in use
- Architecture decisions and patterns
- Project conventions, coding standards
- Dependencies (packages, services, APIs)
- Non-obvious constraints and assumptions

## Tools & Capabilities
- What tools were defined/used?
- Any patterns for tool usage?

## Pending Items
- Explicit next steps the user mentioned
- Implicit next steps based on what was in progress

## Conversation Metadata
- Duration/span of the conversation
- Number of turns compacted
- Pseudo-models used

Format as clean Markdown. Be precise and technical. This is for a developer to continue work — not a generic summary."""
