"""
FastAPI server — receives Jira webhooks and triggers the agent.

Endpoints:
  POST /webhook  → Jira sends ticket/comment events here
  GET  /health   → Simple health check (useful to verify server is running)
"""

import logging
from fastapi import FastAPI, BackgroundTasks
from src.server.models import JiraWebhookPayload
from src.integrations.git_ops import clone_repo, create_branch, commit_changes, push_branch
from src.integrations.github_client import create_pull_request
from src.integrations.jira_client import add_comment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jira Coding Agent")


def process_new_ticket(issue_key: str, summary: str, description: str | None):
    """Handle a new Jira ticket. Runs in the background.

    Phase 1 pipeline:
      Clone → branch → (later: agent changes code) → commit → push → PR → Jira comment

    Phase 2+: The LangGraph agent will slot in between branch and commit,
    making actual code changes based on the ticket.
    """
    try:
        logger.info(f"Processing ticket {issue_key}: {summary}")

        # Step 1: Clone the target repo into workspace/<issue_key>/
        repo_path = clone_repo(issue_key)

        # Step 2: Create a branch named agent/<issue_key>
        branch_name = create_branch(repo_path, issue_key)

        # Step 3: (Phase 2+) LangGraph agent makes code changes here
        # For now, we make a placeholder change so the PR isn't empty
        readme = repo_path / "README.md"
        content = readme.read_text() if readme.exists() else ""
        content += f"\n\n<!-- Agent processing {issue_key}: {summary} -->\n"
        readme.write_text(content)

        # Step 4: Commit the changes
        commit_changes(repo_path, issue_key, summary)

        # Step 5: Push the branch to GitHub
        push_branch(repo_path, branch_name)

        # Step 6: Create a Pull Request
        pr_body = (
            f"## {issue_key}: {summary}\n\n"
            f"**Description:** {description or 'No description provided.'}\n\n"
            f"---\n"
            f"*Automated by Jira Coding Agent*"
        )
        pr_url = create_pull_request(
            branch_name=branch_name,
            title=f"{issue_key}: {summary}",
            body=pr_body,
        )

        # Step 7: Comment the PR link on the Jira ticket
        add_comment(issue_key, f"PR created: {pr_url}")
        logger.info(f"Pipeline complete for {issue_key}: {pr_url}")

    except Exception as e:
        # If anything fails, comment the error on Jira so the human knows
        logger.error(f"Pipeline failed for {issue_key}: {e}")
        try:
            add_comment(issue_key, f"Agent failed: {str(e)}")
        except Exception:
            logger.error(f"Could not comment failure on {issue_key}")


def process_comment(issue_key: str, comment_body: str):
    """Handle a Jira comment. Runs in the background.

    Used for human-in-the-loop: when someone replies "approve" or "reject".
    Phase 1: Just logs the comment.
    Later: This will resume the paused agent.
    """
    logger.info(f"Comment on {issue_key}: {comment_body}")
    # TODO: Resume paused agent if comment is "approve"


@app.post("/webhook")
async def handle_webhook(payload: JiraWebhookPayload, background_tasks: BackgroundTasks):
    """Receive Jira webhook, return 200 immediately, process in background.

    Why background? Jira expects a response within ~10 seconds.
    Our agent takes minutes. If we block, Jira retries and we get
    duplicate PRs. So we acknowledge fast and process async.
    """
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
