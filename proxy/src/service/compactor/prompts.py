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
    return """You are a conversation compactor. Your task is to create a comprehensive, detailed, structured snapshot of a long conversation history.

The conversation is split into THREE pieces:

1. **PRIMER MENSAJE DEL USUARIO** — The very first user message that started the conversation. PRESERVE THIS INTACT in the "Problem State" and "Goal" sections. This is the original context.

2. **HISTORIAL INTERMEDIO** — The middle of the conversation. Compact this into the Technical Decisions, Code Produced, Technical Context, Tools & Capabilities, and Pending Items sections.

3. **ULTIMO MENSAJE DEL USUARIO** — The most recent user message. PRESERVE THIS INTACT in the "Current Status" and "Next Steps" sections. This is the current request.

CRITICAL RULES:
- PRESERVE the first user message's intent and context completely
- PRESERVE the last user message's request and context completely
- YOU MUST compact the middle history between them
- Your output must be at least 5%% of the original conversation length

# Required Sections

## Goal
- What is the overall goal or task? (derived from the FIRST user message)
- Current objective at time of compaction (derived from the LAST user message)

## Problem State
- What problem or task is being worked on?
- What is the current status at the moment of compaction?

## Technical Decisions
- Every significant decision made, WITH its full justification
- Why was approach A chosen over approach B?
- What constraints or tradeoffs influenced each decision?
- Include alternatives that were considered

## Code Produced
- Key code that establishes the current state — include relevant snippets
- Include file paths and full context for each file
- For existing files modified: show the changes and their purpose
- For large files: include key functions, classes, and their signatures
- Include file structure overview

## Current Status
- **Resolved:** completed items with details of what was achieved
- **Unresolved:** pending items with their current state
- **In Progress at compaction:** what was actively being worked on, with context

## Technical Context
- Environment variables in use
- Architecture decisions and patterns
- Project conventions, coding standards
- Dependencies (packages, services, APIs)
- Non-obvious constraints and assumptions
- Commands run and their results

## Tools & Capabilities
- What tools were defined/used?
- Any patterns for tool usage?
- What worked well and what didn't?

## Pending Items
- Explicit next steps the user mentioned
- Implicit next steps based on what was in progress
- Blockers or open questions

## Conversation Metadata
- Duration/span of the conversation
- Number of turns compacted
- Pseudo-models used

Format as clean Markdown with clear sections. Be precise, technical, and THOROUGH. This is for a developer to continue work — the more detail you retain, the better the continuation will be."""
