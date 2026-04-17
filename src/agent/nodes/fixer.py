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
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.state import AgentState
from src.config import secrets

logger = logging.getLogger(__name__)


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

    # Read the test file and any files we previously changed
    file_contents = ""

    # Always include the test file
    test_files = list((repo_path / "src").glob("*.test.*"))
    for tf in test_files:
        relative = str(tf.relative_to(repo_path))
        content = tf.read_text()
        file_contents += f"\n--- {relative} ---\n{content}\n"

    # Include files we changed (they might need fixing too)
    for change in changes_made:
        # change looks like: "src/App.js: replaced 'X' with 'Y'"
        file_path_str = change.split(":")[0].strip()
        full_path = repo_path / file_path_str
        if full_path.exists():
            content = full_path.read_text()
            file_contents += f"\n--- {file_path_str} ---\n{content}\n"

    user_message = (
        f"## Test Failure Output\n```\n{test_output}\n```\n\n"
        f"## Previous Changes Made\n{chr(10).join(f'- {c}' for c in changes_made)}\n\n"
        f"## Relevant Files\n{file_contents}\n\n"
        f"Fix the test failure. The main code change was correct — "
        f"most likely the test file needs to be updated to match the new code."
    )

    llm = ChatGroq(
        api_key=secrets.groq_api_key,
        model="llama-3.3-70b-versatile",
    )
    structured_llm = llm.with_structured_output(FixPlanOutput)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    result = structured_llm.invoke(messages)

    logger.info(f"Fix plan: {result.explanation}")
    for edit in result.edits:
        logger.info(f"  Fix: {edit.file} | '{edit.old_string[:40]}...' → '{edit.new_string[:40]}...'")

    edit_plan = [
        {"file": edit.file, "old_string": edit.old_string, "new_string": edit.new_string}
        for edit in result.edits
    ]

    return {
        "edit_plan": edit_plan,
        "retry_count": retry_count + 1,
    }
