"""
WAIT FOR APPROVAL node — pauses the agent for human review on high-risk changes.

Flow:
  1. Format the agent's plan (parsed ticket + edit plan) into a readable comment
  2. Post it to Jira as a comment with "reply approve/reject" instructions
  3. Call interrupt() — LangGraph pauses and saves the state
  4. Return the comment text — when resumed, this becomes the human's reply

When the human comments "approve" or "reject" on the Jira ticket:
  → comment_created webhook fires
  → app.py finds the paused state and resumes it
  → this node returns, agent continues with WRITE
"""

import logging
from langgraph.types import interrupt
from src.agent.state import AgentState
from src.integrations.jira_client import add_comment

logger = logging.getLogger(__name__)


def wait_for_approval(state: AgentState) -> dict:
    """WAIT FOR APPROVAL node — called by LangGraph for high-risk changes.

    Reads: ticket_plan, edit_plan, issue_key from state
    Writes: approval_status (after resumed)
    """
    issue_key = state["issue_key"]
    ticket_plan = state["ticket_plan"]
    edit_plan = state.get("edit_plan", [])

    logger.info(f"PAUSING for human approval on {issue_key} (HIGH RISK)")

    # Format the plan into a readable Jira comment
    edit_summary = ""
    for edit in edit_plan:
        old_preview = edit["old_string"][:60].replace("\n", " ") + "..."
        new_preview = edit["new_string"][:60].replace("\n", " ") + "..."
        edit_summary += f"- **{edit['file']}**\n  - Replace: `{old_preview}`\n  - With: `{new_preview}`\n"

    comment_text = (
        f"🤖 *AGENT PAUSED — High Risk Change*\n\n"
        f"**Intent:** {ticket_plan['intent']}\n"
        f"**Risk Level:** {ticket_plan['risk_level']}\n\n"
        f"**Planned Edits:**\n{edit_summary}\n"
        f"---\n"
        f"Reply with `approve` to proceed, or `reject` to cancel."
    )

    # Post the plan to Jira
    add_comment(issue_key, comment_text)
    logger.info(f"Posted approval request on {issue_key}, waiting for response...")

    # interrupt() — LangGraph pauses here
    # When resumed, the value passed to Command(resume=...) becomes this return value
    human_response = interrupt(
        {"action": "wait_for_approval", "issue_key": issue_key}
    )

    # When the human replies, app.py calls agent.invoke(Command(resume="approved" or "rejected"))
    # interrupt() returns whatever was passed in resume=...
    logger.info(f"Resumed with response: {human_response}")

    return {"approval_status": human_response}
