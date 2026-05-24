"""Compaction prompts for Sprint 4.

plan-proxy.md §9.2: Pre-compaction prompt extracts relevant info.
plan-proxy.md §10.3: Continuous compaction prompt generates structured snapshot.

Both prompts are pure functions — deterministic, no side effects.
python.md §4: pure functions, no I/O, no state.
"""


def build_pre_compaction_prompt(
    user_content: str,
    target_tokens: int,
) -> str:
    """Build the prompt for the pre-compactor model.

    The compactor extracts relevant information from a long input,
    preserving technical details while removing noise.

    Args:
        user_content: The original user message content to compact.
        target_tokens: Maximum tokens for the compacted output.

    Returns:
        A prompt string for the compactor model.
    """
    return f"""You are a pre-compactor for an expensive reasoning model.

Your job: Extract from the following text ONLY the information relevant for the user's task.
The user's request is embedded in the text below.

Rules:
1. Preserve all technical details: code snippets, error messages, log lines, file paths, version numbers
2. Preserve all constraints and requirements the user mentioned
3. Remove noise: repeated lines, irrelevant stack traces, boilerplate
4. Structure the output: use sections if the input contains multiple topics
5. Target length: approximately {target_tokens} tokens
6. DO NOT add analysis, suggestions, or commentary — just extract and organize

--- INPUT BELOW ---

{user_content}

--- END INPUT ---

Extracted content (max {target_tokens} tokens):"""


def build_continuous_compaction_prompt() -> str:
    """Build the prompt for continuous conversation compaction.

    Generates a structured Markdown snapshot that preserves all critical
    technical context needed to continue the work.

    Returns:
        A system prompt string for the compactor model.
    """
    return """You are a conversation compactor. Your job is to create a structured snapshot of a long conversation history that preserves all critical technical context needed to continue the work.

Extract and organize the following from the conversation:

### State of the Problem
- What is the central problem or task being worked on?
- What is the current status?

### Technical Decisions Made
- Each decision with its justification (not just the outcome)
- Why was approach A chosen over approach B?
- What constraints influenced these decisions?

### Code Produced (key extracts only)
- Only the code that establishes the current state
- Don't include everything — only what's needed to continue
- Include file paths where relevant

### Current State
- Resolved: list of completed items
- Unresolved: list of pending items
- In Progress at compaction time: what was being worked on

### Technical Context
- Environment variables, architecture, dependencies
- Project conventions, coding standards mentioned
- Any non-obvious constraints or assumptions

### Pending Items
- What the user was going to do next
- Any explicit next steps mentioned

Format the output as Markdown. Be concise but complete. The goal is that someone reading this snapshot can continue the conversation without needing the original history."""


def build_explicit_compaction_prompt() -> str:
    """Build the prompt for explicit conversation compaction (Sprint 6).

    More comprehensive than the continuous compaction prompt.
    Generates a structured Markdown snapshot with all required sections
    to continue the work without accessing the original history.

    plan-proxy.md §11.3: Estructura del snapshot generado.
    sprint §3.5: Explicit compaction prompt specification.

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
