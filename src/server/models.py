"""
Pydantic models for Jira webhook payloads.

Jira sends a lot of data in its webhooks — we only model the fields
we actually need. Pydantic ignores extra fields by default, so this
is safe even when Jira adds new fields in the future.
"""

from pydantic import BaseModel


class JiraStatus(BaseModel):
    name: str  # e.g. "To Do", "In Progress", "Done"


class JiraIssueType(BaseModel):
    name: str  # e.g. "Task", "Bug", "Story"


class JiraIssueFields(BaseModel):
    summary: str  # The ticket title
    description: str | None = None  # The ticket body (can be empty)
    status: JiraStatus
    issuetype: JiraIssueType


class JiraIssue(BaseModel):
    key: str  # e.g. "JCA-1"
    fields: JiraIssueFields


class JiraComment(BaseModel):
    body: str  # The comment text (e.g. "approve" or "reject")


class JiraWebhookPayload(BaseModel):
    """Top-level payload from Jira webhook.

    webhookEvent tells us what happened:
    - "jira:issue_created" → new ticket
    - "comment_created"    → someone commented (used for human-in-the-loop)
    """
    webhookEvent: str
    issue: JiraIssue
    comment: JiraComment | None = None  # Only present for comment events
