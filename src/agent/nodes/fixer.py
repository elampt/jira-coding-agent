"""
FIXER node — reads test failure output and generates fix edits.

This is the "self-heal" brain. When tests fail:
  1. It reads the error message from test_output
  2. Sends it to the LLM along with the relevant file contents
  3. LLM figures out what's wrong and outputs fix edits
  4. WRITE node applies the fixes
  5. TEST runs again

Uses the same Pydantic structured output as the PLAN node.
"""

import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.state import AgentState
from src.integrations.git_ops import worktree_changed_files
from src.llm import invoke_structured

logger = logging.getLogger(__name__)

# ~4K tokens of source. Groq's free tier allows 8K tokens/minute in total, and showing the model
# more code than it needs also tempts it to edit things that aren't broken.
MAX_CONTEXT_CHARS = 16000
SOURCE_SUFFIXES = {".js", ".jsx", ".ts", ".tsx"}
# Paths like src/components/Nav.test.js in Jest output (the lookbehind skips node_modules/.../src/x)
_PATH_IN_OUTPUT = re.compile(r"(?<![\w./-])src/[\w./@-]+\.(?:jsx?|tsx?)")


def _is_source_file(repo_path: Path, relative: str) -> bool:
    path = repo_path / relative
    return path.is_file() and path.suffix in SOURCE_SUFFIXES and "node_modules" not in path.parts


def _files_to_show(state: AgentState, repo_path: Path) -> list[str]:
    """Source files worth showing the model, most relevant first, searched recursively.

    Order: files named in the failure output, files the agent already edited, then the files
    the search step found for this ticket. Falls back to every source file under src/.
    """
    candidates = _PATH_IN_OUTPUT.findall(state["test_output"])
    candidates += worktree_changed_files(repo_path)
    candidates += [f["path"] for f in state.get("relevant_files", [])]

    ordered = list(dict.fromkeys(c for c in candidates if _is_source_file(repo_path, c)))
    if ordered:
        return ordered

    everything = (p for p in (repo_path / "src").rglob("*") if p.suffix in SOURCE_SUFFIXES)
    return [
        str(p.relative_to(repo_path))
        for p in sorted(everything)
        if _is_source_file(repo_path, str(p.relative_to(repo_path)))
    ]


class FixInstruction(BaseModel):
    file: str = Field(description="Relative file path to fix")
    old_string: str = Field(description="The EXACT text to find (copy from the code)")
    new_string: str = Field(description="The replacement text that fixes the issue")


class FixPlanOutput(BaseModel):
    edits: list[FixInstruction]
    explanation: str = Field(description="What was wrong and how this fixes it")


SYSTEM_PROMPT = """You are an AI coding agent fixing a test failure in a React codebase.

You previously made code changes based on a Jira ticket, but the tests are now failing.
Given the test error output and the relevant files, figure out what's wrong and fix it.

STEP 1 — DIAGNOSE where the problem is:
- If the error is an IMPORT error (module not found, cannot resolve) → the MAIN CODE has a bad import. Fix the main code by removing or correcting the import.
- If the error is a SYNTAX error (unexpected token, parsing error) → the MAIN CODE has broken syntax. Fix the main code.
- If the error is a TEST ASSERTION failure (expected X, received Y / element not found) → the TEST FILE needs updating to match the new code.

STEP 2 — FIX the right file:
- Import/syntax errors → fix src/*.js (the main code files)
- Assertion errors → fix src/*.test.js (the test files)
- NEVER keep editing the test file if the main code is the source of the error

Common issues:
- Test expects old text that was changed (update the test to match new text)
- CSS class name changed but test still references old name
- Main code imports a module that doesn't exist (remove or fix the import)
- Main code has syntax errors from bad edits (fix the syntax)

Rules:
1. old_string must be EXACTLY as it appears in the code
2. Make MINIMUM changes to fix the issue
3. If a library/module was imported but doesn't exist in the project, REMOVE the import entirely
4. Do NOT create new files — only edit existing files
5. IMPORTANT: If the same old text appears in MULTIPLE places in a file, create a SEPARATE edit for EACH occurrence. Include enough surrounding context in old_string to make each edit unique. For example, instead of just "learn react", use "renders learn react link" for the first occurrence and "getByText(/learn react/i)" for the second."""


def fix_test_failure(state: AgentState) -> dict:
    """FIXER node — called by LangGraph when tests fail.

    Reads: test_output, changes_made, repo_path from state
    Writes: edit_plan (new fixes), increments retry_count
    """
    test_output = state["test_output"]
    changes_made = state.get("changes_made", [])
    repo_path = Path(state["repo_path"])
    retry_count = state.get("retry_count", 0)

    logger.info(f"Fixing test failure (attempt {retry_count + 1}/3)")

    # Read ALL relevant source files DIRECTLY FROM DISK — not from state.
    # This is critical: after fix attempt 1 modifies files, attempt 2 needs
    # to see the CURRENT state of files, not the original state.
    file_contents = ""
    included = []
    for relative in _files_to_show(state, repo_path):
        content = (repo_path / relative).read_text()
        # Always include the first (most relevant) file; after that, stay inside the budget.
        if included and len(file_contents) + len(content) > MAX_CONTEXT_CHARS:
            continue
        included.append(relative)
        file_contents += f"\n--- {relative} ---\n{content}\n"
    logger.info(f"Fixer context: {len(included)} file(s): {included}")

    user_message = (
        f"## Test Failure Output\n```\n{test_output}\n```\n\n"
        f"## Previous Changes Made\n{chr(10).join(f'- {c}' for c in changes_made)}\n\n"
        f"## Current Source Files (read from disk)\n{file_contents}\n\n"
        f"Fix the test failure. Diagnose whether the problem is in the main "
        f"code or the test file, then fix the right file."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    result = invoke_structured(FixPlanOutput, messages)

    logger.info(f"Fix plan: {result.explanation}")
    for edit in result.edits:
        logger.info(
            f"  Fix: {edit.file} | '{edit.old_string[:40]}...' → '{edit.new_string[:40]}...'"
        )

    edit_plan = [
        {"file": edit.file, "old_string": edit.old_string, "new_string": edit.new_string}
        for edit in result.edits
    ]

    return {
        "edit_plan": edit_plan,
        "retry_count": retry_count + 1,
    }
