"""
Observability — LangFuse integration for tracing LLM calls and agent flow.

LangFuse captures:
  - Every LLM call: prompt, response, latency, tokens, cost
  - Every LangGraph node execution: name, input state, output state, duration
  - Full agent traces: see the entire flow as a tree in the LangFuse UI

Usage:
  config={"callbacks": [langfuse_handler], "configurable": {"thread_id": ...}}

When LANGFUSE keys are not set in .env, this returns None and tracing is
silently skipped — useful for development without LangFuse setup.
"""

import logging

from src.config import config, secrets

logger = logging.getLogger(__name__)


def get_langfuse_handler():
    """Return a configured LangFuse callback handler, or None if not configured.

    Returns None instead of raising — so missing LangFuse keys don't break the agent.
    """
    if not secrets.langfuse_public_key or not secrets.langfuse_secret_key:
        logger.info("LangFuse keys not configured — tracing disabled")
        return None

    try:
        # Initialize the Langfuse client first so it picks up env vars
        import os

        from langfuse.langchain import CallbackHandler

        os.environ["LANGFUSE_PUBLIC_KEY"] = secrets.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = secrets.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = secrets.langfuse_host

        handler = CallbackHandler()
        logger.info(f"LangFuse tracing enabled → {secrets.langfuse_host}")
        return handler
    except Exception as e:
        logger.warning(f"Failed to initialize LangFuse: {e}")
        return None


def trace_metadata(
    issue_key: str, environment: str = "prod", extra_tags: tuple[str, ...] = ()
) -> dict:
    """Labels LangFuse attaches to a run: session = ticket key, a readable trace name, tags.

    The session id ties a paused run and its later resumed half into one story. `environment`
    ("prod" for real tickets, "eval" for harness runs) keeps the two apart when filtering.
    """
    return {
        "langfuse_session_id": issue_key,
        "langfuse_trace_name": "jira-agent-run",
        "langfuse_tags": [environment, f"model:{config.llm.model}", *extra_tags],
    }


OUTCOME_SUCCESS = "success"
OUTCOME_AWAITING_APPROVAL = "awaiting_approval"
OUTCOME_REJECTED = "rejected"
OUTCOME_ENVIRONMENT_FAILURE = "environment_failure"
OUTCOME_TESTS_FAILING = "tests_failing"
OUTCOME_ERROR = "error"


def classify_outcome(
    *,
    error: bool,
    paused: bool,
    approval_status: str | None,
    environment_failure: bool,
    tests_passed: bool,
) -> str:
    """Map a finished run's final state to one outcome label (used by the eval harness)."""
    if error:
        return OUTCOME_ERROR
    if approval_status == "rejected":
        return OUTCOME_REJECTED
    if environment_failure:
        return OUTCOME_ENVIRONMENT_FAILURE
    if tests_passed:
        return OUTCOME_SUCCESS
    return OUTCOME_AWAITING_APPROVAL if paused else OUTCOME_TESTS_FAILING


def record_run_outcome(issue_key: str, outcome: str, risk_level: str | None = None) -> None:
    """Attach how a ticket's run ended (and its risk level) to its LangFuse session.

    Scores use a fixed id per ticket, so a ticket that goes awaiting_approval -> success updates
    one score instead of leaving two. Tracing must never break a ticket, so this never raises.
    """
    if langfuse_handler is None:
        return
    try:
        from langfuse import get_client

        client = get_client()
        client.create_score(
            name="outcome",
            value=outcome,
            session_id=issue_key,
            score_id=f"outcome-{issue_key}",
            data_type="CATEGORICAL",
        )
        if risk_level:
            client.create_score(
                name="risk_level",
                value=str(risk_level).lower(),
                session_id=issue_key,
                score_id=f"risk-{issue_key}",
                data_type="CATEGORICAL",
            )
    except Exception as e:
        logger.warning(f"Could not record outcome for {issue_key} in LangFuse: {e}")


# Singleton — created once when the module is imported
langfuse_handler = get_langfuse_handler()
