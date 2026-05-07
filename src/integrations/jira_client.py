"""
Jira client — talks TO Jira (opposite direction of the webhook).

Webhook:      Jira → Our server  (Jira notifies us)
This client:  Our server → Jira  (We update tickets, add comments)

Uses atlassian-python-api which handles authentication and HTTP calls.
We just call methods like jira.issue("ELAMP2-1") and get back data.
"""

import logging

from atlassian import Jira

from src.config import secrets

logger = logging.getLogger(__name__)


def create_jira_client() -> Jira:
    """Create an authenticated Jira client using credentials from .env.

    Creates a new client per call — simpler and always uses latest credentials.
    For our throughput (one ticket at a time), the overhead is negligible.
    """
    return Jira(
        url=secrets.jira_base_url,
        username=secrets.jira_email,
        password=secrets.jira_api_token,  # API token goes here, not actual password
    )


def get_issue(issue_key: str) -> dict:
    """Read full ticket details from Jira.

    Returns a dict with keys like 'key', 'fields' (summary, description, status, etc.)
    We use this when the agent needs the full ticket context beyond what the webhook sent.
    """
    jira = create_jira_client()
    issue = jira.issue(issue_key)
    logger.info(f"Fetched issue {issue_key}: {issue['fields']['summary']}")
    return issue


def add_comment(issue_key: str, comment_text: str) -> None:
    """Post a comment on a Jira ticket.

    Used for:
    - "PR created: <link>" after creating a PR
    - Posting the agent's plan for human approval
    - "Couldn't fix, needs human" when self-heal fails
    """
    jira = create_jira_client()
    jira.issue_add_comment(issue_key, comment_text)
    logger.info(f"Added comment to {issue_key}")


def update_status(issue_key: str, status_name: str) -> None:
    """Change a ticket's status (e.g., 'To Do' → 'In Progress').

    Jira uses 'transitions' — you can't directly set a status.
    Each status change follows an allowed path (like a state machine):
      To Do → In Progress → In Review → Done

    So we:
    1. Ask Jira: "what transitions are available for this ticket?"
    2. Find the one matching our target status name
    3. Execute that transition
    """
    jira = create_jira_client()

    # Get available transitions for this ticket
    transitions = jira.get_issue_transitions(issue_key)

    # Find the transition that leads to our target status
    target_transition = None
    for t in transitions:
        if t["name"].lower() == status_name.lower():
            target_transition = t
            break

    if target_transition is None:
        logger.warning(
            f"No transition found for status '{status_name}' on {issue_key}. "
            f"Available: {[t['name'] for t in transitions]}"
        )
        return

    jira.set_issue_status(issue_key, target_transition["name"])
    logger.info(f"Updated {issue_key} status to '{status_name}'")
