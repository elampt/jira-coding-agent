"""
PLAN node — LLM reads ticket + code and outputs precise edit instructions.

Input:  ticket_plan + relevant_files from state
Output: edit_plan — list of {file, old_string, new_string}

The LLM sees the actual code and creates surgical edits.
Uses old_string/new_string (not line numbers) because line numbers
are fragile — they shift when code changes. String matching is reliable.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.agent.state import AgentState
from src.config import secrets

logger = logging.getLogger(__name__)


# --- Pydantic models for structured LLM output ---


class EditInstruction(BaseModel):
    """A single edit the agent should make."""

    file: str = Field(description="Relative file path, e.g. src/App.js")
    old_string: str = Field(
        description="The EXACT text to find in the file (copy-paste from the code)"
    )
    new_string: str = Field(description="The replacement text")


class EditPlanOutput(BaseModel):
    """Complete edit plan — list of all edits needed."""

    edits: list[EditInstruction] = Field(description="All edits to make, in order")
    explanation: str = Field(description="Brief explanation of what changes are being made and why")


# --- System prompt ---

SYSTEM_PROMPT = """You are an AI coding agent that modifies a React frontend codebase based on Jira tickets.

Given a ticket and the relevant source files, create a precise edit plan.

Rules:
1. old_string must be EXACTLY as it appears in the code — copy it character for character, including whitespace and quotes
2. new_string is what replaces it
3. Make the MINIMUM changes needed to fulfill the ticket. Do not refactor, clean up, or "improve" surrounding code
4. Only edit files that were provided to you. Do not create new files
5. If the ticket asks to change text, only change that text — don't restructure the component
6. If the ticket asks to change a color/style, only change that specific CSS property
7. IMPORTANT: If your changes modify text, class names, component props, or any value that is referenced in test files (*.test.js), include edits to update those test files in the SAME plan. Tests should pass after your edits without needing a separate fix.

8. Only import from packages that ALREADY exist in the project. Built-in React hooks (useState, useEffect, etc.) are fine. Do NOT add imports from new npm packages that aren't already in the codebase — they won't be installed and the code will crash.
9. To ADD a new import, use the existing first import line as old_string and include the new import in new_string. Example:
   old_string: "import logo from './logo.svg';"
   new_string: "import { useState } from 'react';\nimport logo from './logo.svg';"
   Do NOT use an empty old_string — always anchor to existing code.
10. PREFER FEWER, LARGER edits over many small ones. If you need to add state + JSX + handler to a component, replace the ENTIRE function body in ONE edit rather than making 5 separate surgical edits. Each edit that depends on a previous edit is a point of failure. One large replacement is safer than a chain of small ones.
11. For test files: replace the ENTIRE test function body in ONE edit, not individual lines. This prevents duplicate variable declarations and structural corruption.

CRITICAL: The old_string must match EXACTLY. If you get even one character wrong, the edit will fail."""


# --- Node function ---


def plan_changes(state: AgentState) -> dict:
    """PLAN node — called by LangGraph.

    Reads: ticket_plan, relevant_files from state
    Writes: edit_plan to state
    """
    ticket_plan = state["ticket_plan"]
    relevant_files = state["relevant_files"]

    logger.info(f"Planning changes for: {ticket_plan['intent']}")

    # Build the prompt with ticket info + actual code
    file_contents = ""
    for f in relevant_files:
        file_contents += f"\n--- {f['path']} ---\n{f['content']}\n"

    user_message = (
        f"## Ticket\n"
        f"Intent: {ticket_plan['intent']}\n"
        f"Summary: {state['summary']}\n"
        f"Description: {state.get('description', 'No description')}\n\n"
        f"## Source Files\n{file_contents}\n\n"
        f"Create an edit plan to fulfill this ticket."
    )

    # Create LLM with structured output
    llm = ChatGroq(
        api_key=secrets.groq_api_key,
        model="llama-3.3-70b-versatile",
    )
    structured_llm = llm.with_structured_output(EditPlanOutput)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    result = structured_llm.invoke(messages)

    logger.info(f"Plan: {result.explanation}")
    for edit in result.edits:
        logger.info(
            f"  Edit: {edit.file} | '{edit.old_string[:40]}...' → '{edit.new_string[:40]}...'"
        )

    # Convert to list of dicts for state
    edit_plan = [
        {"file": edit.file, "old_string": edit.old_string, "new_string": edit.new_string}
        for edit in result.edits
    ]

    return {"edit_plan": edit_plan}
