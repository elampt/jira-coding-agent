"""
AgentState — the data that flows between LangGraph nodes.

Each node reads what it needs from state, does its work,
and returns a partial update with only the fields it changed.
LangGraph merges the update into the full state automatically.

Example flow:
  PARSE reads summary/description → writes ticket_plan
  SEARCH reads ticket_plan → writes relevant_files
  PLAN reads ticket_plan + relevant_files → writes edit_plan
  WRITE reads edit_plan → writes changes_made
"""

from typing import TypedDict


class TicketPlan(TypedDict):
    """Output of PARSE node — structured understanding of the ticket."""
    intent: str                  # What the ticket wants: "change text", "change color"
    component_hints: list[str]   # Keywords to search for: ["header", "login", "button"]
    risk_level: str              # "low", "medium", "high"


class FileContext(TypedDict):
    """A relevant file found by SEARCH node."""
    path: str       # Relative path: "src/App.js"
    content: str    # Full file content


class EditInstruction(TypedDict):
    """A single edit planned by PLAN node."""
    file: str       # Which file to edit: "src/App.js"
    old_string: str  # The exact text to find
    new_string: str  # What to replace it with


class AgentState(TypedDict):
    """Full state passed between all nodes.

    Nodes return partial updates — only the fields they changed.
    LangGraph merges them into this full state automatically.
    """
    # Input — set once at the start, never changed
    issue_key: str
    summary: str
    description: str
    repo_path: str
    branch_name: str

    # PARSE node output
    ticket_plan: TicketPlan

    # SEARCH node output
    relevant_files: list[FileContext]

    # PLAN node output
    edit_plan: list[EditInstruction]

    # WRITE node output
    changes_made: list[str]

    # TEST node output
    test_passed: bool              # Did tests pass?
    test_output: str               # stdout/stderr from npm test
    retry_count: int               # How many times we've retried (max 3)

    # SCREENSHOT node outputs
    screenshot_before: str         # Path to "before" screenshot (empty if failed)
    screenshot_after: str          # Path to "after" screenshot (empty if failed)

    # Human-in-the-loop
    approval_status: str           # "pending" / "approved" / "rejected"
