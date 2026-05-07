"""
WAIT FOR APPROVAL node — pauses the agent for human review on high-risk changes.

Flow:
  1. Generate an LLM summary of the planned edits (human-readable)
  2. Post structured plan to Jira: summary + affected files + risk concerns + detailed edits
  3. Call interrupt() — LangGraph pauses and saves state
  4. When resumed via Command(resume=...), interrupt() returns the human's response

The LLM summary makes the comment actually useful for reviewers — they can
understand the change at a glance instead of mentally diffing truncated strings.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.agent.state import AgentState
from src.config import secrets
from src.integrations.jira_client import add_comment

logger = logging.getLogger(__name__)

# Tracks which tickets already have their approval comment posted.
# Prevents duplicate comments when LangGraph re-runs the node on resume.
_posted_approvals: set[str] = set()


class ChangeSummary(BaseModel):
    """LLM-generated summary of the agent's plan, for human review."""

    summary: str = Field(
        description="2-3 sentences in plain English explaining what the agent is about to do and why"
    )
    risk_concerns: list[str] = Field(
        description="Specific things the reviewer should worry about — side effects, dependencies, breaking changes. 2-4 items."
    )


SUMMARY_SYSTEM_PROMPT = """You are reviewing an AI coding agent's planned changes and writing a summary for a human reviewer.

Your job: convert technical edits into a clear, plain-English explanation that a reviewer can quickly understand.

Be specific about:
- What functional change is being made (not just "edits App.js" — say "adds OAuth login button")
- What might go wrong (new dependencies, breaking changes, test impacts)

Be concise. Reviewers are busy — short and clear beats long and thorough."""


def _generate_summary(
    ticket_plan: dict, edit_plan: list[dict], summary: str, description: str
) -> ChangeSummary:
    """Ask the LLM to summarize the planned changes for human review."""
    # Build a clear prompt with all context
    edits_text = ""
    for i, edit in enumerate(edit_plan, 1):
        edits_text += (
            f"\n### Edit {i}: {edit['file']}\n"
            f"**Old:**\n```\n{edit['old_string']}\n```\n"
            f"**New:**\n```\n{edit['new_string']}\n```\n"
        )

    user_message = (
        f"## Jira Ticket\n"
        f"Title: {summary}\n"
        f"Description: {description}\n"
        f"Intent (parsed): {ticket_plan['intent']}\n\n"
        f"## Planned Edits\n{edits_text}\n\n"
        f"Generate a summary of what this agent is about to do, "
        f"plus 2-4 specific risks the reviewer should watch for."
    )

    llm = ChatGroq(api_key=secrets.groq_api_key, model="llama-3.3-70b-versatile")
    structured_llm = llm.with_structured_output(ChangeSummary)
    return structured_llm.invoke(  # pyright: ignore[reportReturnType]
        [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )


def _format_jira_comment(
    summary_obj: ChangeSummary,
    ticket_plan: dict,
    edit_plan: list[dict],
) -> str:
    """Format the approval request as a Jira comment.

    Structure:
      1. Header (PAUSED + risk level)
      2. LLM summary (the high-value bit)
      3. Risk concerns (what to watch for)
      4. Affected files summary (which files, how many edits)
      5. Detailed edits (collapsible style — for reviewers who want details)
      6. Approval instructions
    """
    # Affected files summary
    files_count: dict[str, int] = {}
    for edit in edit_plan:
        files_count[edit["file"]] = files_count.get(edit["file"], 0) + 1
    files_summary = ", ".join(
        f"{f} ({n} edit{'s' if n > 1 else ''})" for f, n in files_count.items()
    )

    # Risk concerns as bullets
    risk_bullets = "\n".join(f"- {r}" for r in summary_obj.risk_concerns)

    # Detailed edits — full strings, not truncated
    detailed_edits = ""
    for i, edit in enumerate(edit_plan, 1):
        detailed_edits += (
            f"\n*Edit {i} — {edit['file']}*\n"
            f"{{code}}\n"
            f"OLD:\n{edit['old_string']}\n\n"
            f"NEW:\n{edit['new_string']}\n"
            f"{{code}}\n"
        )

    return (
        f"🤖 *AGENT PAUSED — High Risk Change*\n\n"
        f"📝 *What I'm planning to do:*\n"
        f"{summary_obj.summary}\n\n"
        f"⚠️ *Risk Concerns:*\n"
        f"{risk_bullets}\n\n"
        f"📁 *Affected Files:* {files_summary}\n\n"
        f"🔍 *Detailed Edits:*\n"
        f"{detailed_edits}\n"
        f"---\n"
        f"Reply with `approve` to proceed, or `reject` to cancel."
    )


def wait_for_approval(state: AgentState) -> dict:
    """WAIT FOR APPROVAL node — called by LangGraph for high-risk changes.

    IMPORTANT: LangGraph re-runs this entire function from the top when resuming.
    Everything above interrupt() runs TWICE — once on initial call, once on resume.
    We use _posted_approvals (module-level set) to track which tickets already
    have their approval comment posted, so we don't post duplicates.

    Reads: ticket_plan, edit_plan, issue_key, summary, description from state
    Writes: approval_status (after resumed via Command)
    """
    issue_key = state["issue_key"]

    # Only post the approval comment ONCE per ticket.
    # On resume, this block is skipped because issue_key is already in the set.
    if issue_key not in _posted_approvals:
        ticket_plan = state["ticket_plan"]
        edit_plan = state.get("edit_plan", [])
        summary = state["summary"]
        description = state.get("description", "")

        logger.info(f"PAUSING for human approval on {issue_key} (HIGH RISK)")

        # Generate plain-English summary using LLM
        logger.info("Generating change summary for human review...")
        try:
            summary_obj = _generate_summary(ticket_plan, edit_plan, summary, description)
            logger.info(f"Summary: {summary_obj.summary}")
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}. Using fallback.")
            summary_obj = ChangeSummary(
                summary=f"Agent plans to make {len(edit_plan)} edit(s) for: {ticket_plan['intent']}",
                risk_concerns=[
                    "LLM-generated summary unavailable — review the detailed edits below carefully"
                ],
            )

        # Format and post to Jira
        comment_text = _format_jira_comment(summary_obj, ticket_plan, edit_plan)
        add_comment(issue_key, comment_text)
        logger.info(f"Posted approval request on {issue_key}, waiting for response...")

        # Mark as posted so we don't post again on resume
        _posted_approvals.add(issue_key)

    # interrupt() pauses the graph on first call.
    # On resume, it returns the value from Command(resume=...).
    human_response = interrupt({"action": "wait_for_approval", "issue_key": issue_key})

    logger.info(f"Resumed with response: {human_response}")

    # Clean up the tracking set
    _posted_approvals.discard(issue_key)

    return {"approval_status": human_response}
