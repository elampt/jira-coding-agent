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

from src.config import secrets

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


# Singleton — created once when the module is imported
langfuse_handler = get_langfuse_handler()
