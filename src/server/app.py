"""
FastAPI server — receives Jira webhooks and triggers the agent.

Endpoints:
  POST /webhook  → Jira sends ticket/comment events here
  GET  /health   → Simple health check (useful to verify server is running)

The pipeline has two entry points:
  1. New ticket → start agent (may pause at wait_approval)
  2. Approve/reject comment → resume paused agent

Because resuming happens later in a different request, we store per-ticket
context (repo_path, branch_name, screenshot paths) in _session_store so the
"after resume" steps can continue where the initial run left off.
"""

import logging
import subprocess
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks
from langgraph.types import Command

from src.server.models import JiraWebhookPayload
from src.integrations.git_ops import clone_repo, create_branch, commit_changes, push_branch
from src.integrations.github_client import create_pull_request
from src.integrations.jira_client import add_comment
from src.rag.indexer import index_repo
from src.agent.graph import agent
from src.agent.nodes.screenshotter import _capture_screenshot
from src.observability import langfuse_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jira Coding Agent")

# In-memory session store: issue_key → {repo_path, branch_name, before_path, summary, description}
# Needed because approval may arrive in a later webhook — we need to remember context.
# For production, this would be a database. MemorySaver in graph.py handles the agent state.
_session_store: dict[str, dict] = {}


def _agent_config(issue_key: str) -> dict:
    """Build the LangGraph config for a given ticket.

    Includes:
      - thread_id: lets LangGraph match this run with any previously-paused state
      - callbacks: LangFuse handler if configured, so all LLM calls are traced
    """
    config = {"configurable": {"thread_id": issue_key}}
    if langfuse_handler is not None:
        config["callbacks"] = [langfuse_handler]
    return config


def _finalize(issue_key: str, result: dict) -> None:
    """Runs AFTER the agent completes (either normally or after approval).

    Handles: test results check, after screenshot, commit, push, PR, Jira comment.
    """
    session = _session_store.get(issue_key)
    if not session:
        logger.error(f"No session found for {issue_key} — cannot finalize")
        return

    repo_path = Path(session["repo_path"])
    branch_name = session["branch_name"]
    summary = session["summary"]
    description = session["description"]
    before_path = Path(session["before_path"])

    changes_made = result.get("changes_made", [])
    test_passed = result.get("test_passed", False)
    retry_count = result.get("retry_count", 0)
    approval_status = result.get("approval_status", "")

    # If the human rejected, stop here
    if approval_status == "rejected":
        add_comment(issue_key, "🤖 Task cancelled by user.")
        logger.info(f"Task cancelled for {issue_key}")
        _session_store.pop(issue_key, None)
        return

    # If tests still fail after all retries, stop
    if not test_passed:
        add_comment(
            issue_key,
            f"Agent made changes but tests are still failing after {retry_count} fix attempts. "
            f"Needs human review.\n\nTest output:\n```\n{result.get('test_output', '')[-500:]}\n```"
        )
        logger.warning(f"Tests still failing for {issue_key} — commented on Jira")
        _session_store.pop(issue_key, None)
        return

    # Take "after" screenshot — saved directly in the repo (same folder as "before")
    after_path = repo_path / "agent-screenshots" / "after.png"
    _capture_screenshot(repo_path, after_path, "AFTER")

    # Commit + push + PR
    commit_changes(repo_path, issue_key, summary)
    push_branch(repo_path, branch_name)

    # Build PR description with embedded images
    # GitHub renders ![alt](path) as inline images when the file exists in the branch
    changes_list = "\n".join(f"- {c}" for c in changes_made) if changes_made else "No changes made"
    test_info = "All tests passing"
    if retry_count > 0:
        test_info += f" (fixed after {retry_count} retry attempts)"

    screenshot_info = ""
    if before_path.exists():
        screenshot_info += "**Before:**\n\n![Before](agent-screenshots/before.png)\n\n"
    if after_path.exists():
        screenshot_info += "**After:**\n\n![After](agent-screenshots/after.png)\n\n"

    pr_body = (
        f"## {issue_key}: {summary}\n\n"
        f"**Description:** {description or 'No description provided.'}\n\n"
        f"### Changes Made\n{changes_list}\n\n"
        f"### Test Results\n{test_info}\n\n"
    )
    if screenshot_info:
        pr_body += f"### Screenshots\n{screenshot_info}\n"
    pr_body += "---\n*Automated by Jira Coding Agent*"

    pr_url = create_pull_request(
        branch_name=branch_name,
        title=f"{issue_key}: {summary}",
        body=pr_body,
    )
    add_comment(issue_key, f"PR created: {pr_url}")
    logger.info(f"Pipeline complete for {issue_key}: {pr_url}")

    # Cleanup session
    _session_store.pop(issue_key, None)


def process_new_ticket(issue_key: str, summary: str, description: str | None):
    """Handle a new Jira ticket. Runs in the background.

    May pause at wait_approval if the ticket is high-risk.
    When paused, _session_store keeps context for the later resume.
    """
    try:
        logger.info(f"Processing ticket {issue_key}: {summary}")

        repo_path = clone_repo(issue_key)

        logger.info("Installing npm dependencies...")
        subprocess.run(["npm", "install"], cwd=str(repo_path), capture_output=True, timeout=120)

        index_repo(repo_path)

        # Take "before" screenshot — saved directly inside the cloned repo
        # so it gets committed with the PR branch (no separate copy step)
        screenshots_in_repo = repo_path / "agent-screenshots"
        screenshots_in_repo.mkdir(exist_ok=True)
        before_path = screenshots_in_repo / "before.png"
        _capture_screenshot(repo_path, before_path, "BEFORE")

        branch_name = create_branch(repo_path, issue_key)

        # Store session — needed if agent pauses or when finalizing
        _session_store[issue_key] = {
            "repo_path": str(repo_path),
            "branch_name": branch_name,
            "before_path": str(before_path),
            "summary": summary,
            "description": description or "",
        }

        # Run the agent — use issue_key as thread_id so we can resume later
        # graph_config also includes the LangFuse callback if configured
        graph_config = _agent_config(issue_key)
        result = agent.invoke(
            {
                "issue_key": issue_key,
                "summary": summary,
                "description": description or "",
                "repo_path": str(repo_path),
                "branch_name": branch_name,
            },
            config=graph_config,
        )

        # Check if the agent paused at wait_approval
        # When interrupted, LangGraph returns a state where the next node is "wait_approval"
        state_snapshot = agent.get_state(graph_config)
        if state_snapshot.next:
            logger.info(
                f"Agent paused on {issue_key} at node {state_snapshot.next} — waiting for human reply"
            )
            return

        # Agent ran to completion — finalize
        _finalize(issue_key, result)

    except Exception as e:
        logger.error(f"Pipeline failed for {issue_key}: {e}", exc_info=True)
        try:
            add_comment(issue_key, f"Agent failed: {str(e)}")
        except Exception:
            logger.error(f"Could not comment failure on {issue_key}")
        _session_store.pop(issue_key, None)


def process_comment(issue_key: str, comment_body: str):
    """Handle a Jira comment. Runs in the background.

    Used for human-in-the-loop: when someone replies "approve" or "reject".
    Resumes the paused agent with the human's response.
    """
    logger.info(f"Comment on {issue_key}: {comment_body}")

    # Only care about comments if we have a paused session
    if issue_key not in _session_store:
        logger.info(f"No paused session for {issue_key} — ignoring comment")
        return

    # Skip agent's own comments — they all start with "🤖"
    # Without this, the agent's own "Reply with `approve`" message would trigger a resume
    if comment_body.lstrip().startswith("🤖"):
        logger.info(f"Skipping agent's own comment")
        return

    # Detect intent — be strict to avoid false positives.
    # The cleaned comment must be EXACTLY "approve"/"reject" (with optional whitespace),
    # ignoring any Jira account-mention prefix like "[~accountid:...]".
    import re
    cleaned = re.sub(r"\[~accountid:[^\]]+\]", "", comment_body).strip().lower()

    if cleaned in ("approve", "approved"):
        human_response = "approved"
    elif cleaned in ("reject", "rejected"):
        human_response = "rejected"
    else:
        logger.info(f"Comment is not a clear approval/rejection — ignoring")
        return

    logger.info(f"Resuming agent for {issue_key} with: {human_response}")

    try:
        graph_config = _agent_config(issue_key)
        result = agent.invoke(Command(resume=human_response), config=graph_config)
        _finalize(issue_key, result)

    except Exception as e:
        logger.error(f"Resume failed for {issue_key}: {e}", exc_info=True)
        try:
            add_comment(issue_key, f"Agent failed while resuming: {str(e)}")
        except Exception:
            pass
        _session_store.pop(issue_key, None)


@app.post("/webhook")
async def handle_webhook(payload: JiraWebhookPayload, background_tasks: BackgroundTasks):
    """Receive Jira webhook, return 200 immediately, process in background."""
    event = payload.webhookEvent
    issue_key = payload.issue.key
    summary = payload.issue.fields.summary

    logger.info(f"Webhook received: {event} for {issue_key}")

    if event == "jira:issue_created":
        background_tasks.add_task(
            process_new_ticket,
            issue_key=issue_key,
            summary=summary,
            description=payload.issue.fields.description,
        )
        return {"status": "accepted", "issue": issue_key, "action": "processing"}

    elif event == "comment_created" and payload.comment:
        background_tasks.add_task(
            process_comment,
            issue_key=issue_key,
            comment_body=payload.comment.body,
        )
        return {"status": "accepted", "issue": issue_key, "action": "comment_received"}

    else:
        logger.info(f"Ignoring event: {event}")
        return {"status": "ignored", "event": event}


@app.get("/health")
async def health_check():
    """Simple health check — verify the server is running."""
    return {"status": "ok"}
