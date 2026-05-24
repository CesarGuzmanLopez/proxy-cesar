"""Tests for compaction prompts.

Sprint 4 §5.3 — minimum 5 tests.
Verifies that prompts are well-formed, complete, and deterministic.
"""

from src.service.compactor.prompts import build_continuous_compaction_prompt, build_pre_compaction_prompt


def test_pre_compaction_prompt_contains_user_content():
    """Pre-compaction prompt includes the original user content."""
    prompt = build_pre_compaction_prompt(
        user_content="Fix the database connection timeout in src/db/connection.ts",
        target_tokens=8000,
    )
    assert "src/db/connection.ts" in prompt
    assert "database connection timeout" in prompt


def test_pre_compaction_prompt_specifies_target_tokens():
    """Pre-compaction prompt specifies the target token count."""
    prompt = build_pre_compaction_prompt(
        user_content="Some long text here",
        target_tokens=4000,
    )
    assert "4000" in prompt
    assert "tokens" in prompt.lower()


def test_continuous_compaction_prompt_asks_structured_output():
    """Continuous compaction prompt asks for structured Markdown output."""
    prompt = build_continuous_compaction_prompt()
    assert "State of the Problem" in prompt
    assert "Technical Decisions Made" in prompt
    assert "Code Produced" in prompt
    assert "Current State" in prompt
    assert "Technical Context" in prompt
    assert "Pending Items" in prompt
    assert "Markdown" in prompt or "markdown" in prompt


def test_continuous_compaction_prompt_all_required_sections():
    """Continuous compaction prompt covers all required sections from the spec."""
    prompt = build_continuous_compaction_prompt()
    required_sections = [
        "State of the Problem",
        "Technical Decisions Made",
        "Code Produced",
        "Current State",
        "Technical Context",
        "Pending Items",
    ]
    for section in required_sections:
        assert section in prompt, f"Missing required section: {section}"


def test_prompts_are_deterministic():
    """Both prompts are pure functions — same input → same output."""
    user_content = "Test content for deterministic check."

    p1 = build_pre_compaction_prompt(user_content=user_content, target_tokens=8000)
    p2 = build_pre_compaction_prompt(user_content=user_content, target_tokens=8000)
    assert p1 == p2

    c1 = build_continuous_compaction_prompt()
    c2 = build_continuous_compaction_prompt()
    assert c1 == c2


def test_pre_compaction_prompt_empty_content():
    """Pre-compaction prompt handles empty user content gracefully."""
    prompt = build_pre_compaction_prompt(user_content="", target_tokens=8000)
    assert "Extracted content" in prompt
    assert "--- INPUT BELOW ---" in prompt


def test_pre_compaction_prompt_rules_present():
    """Pre-compaction prompt includes all extraction rules."""
    prompt = build_pre_compaction_prompt(user_content="test", target_tokens=8000)
    rules = [
        "Preserve all technical details",
        "Preserve all constraints",
        "Remove noise",
        "Structure the output",
        "DO NOT add analysis",
    ]
    for rule in rules:
        assert rule in prompt, f"Missing rule: {rule}"
