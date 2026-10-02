"""
LLM factory — the one place that turns config.yaml's `llm` section into a client.

Every node that needs an LLM goes through here instead of constructing a client
inline, so swapping the model is a one-line change in config.yaml.
"""

import logging
from typing import TypeVar, cast

import groq
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from src.config import config, secrets

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_STRUCTURED_ATTEMPTS = 3

# Groq 400s that are sampling hiccups (the model emitted a malformed or unregistered tool call),
# not bad requests — an identical retry usually succeeds. Rate limits, 5xx and connection errors
# are already retried by the Groq SDK itself.
_RETRYABLE_MARKERS = ("tool_use_failed", "json_validate_failed")


class LLMCallError(RuntimeError):
    """The model kept returning an unusable response. Message is short enough for a Jira comment."""


def get_llm() -> ChatGroq:
    """Build the chat model described by config.llm (provider + model)."""
    provider = config.llm.provider.lower()
    if provider == "groq":
        return ChatGroq(api_key=secrets.groq_api_key, model=config.llm.model)
    raise ValueError(
        f"Unsupported llm.provider {config.llm.provider!r} in config.yaml. "
        "Only 'groq' is implemented — add a branch in src/llm.py to support another provider."
    )


def _short_reason(error: groq.BadRequestError) -> str:
    body = error.body if isinstance(error.body, dict) else {}
    inner = body.get("error", body)
    text = inner.get("message") if isinstance(inner, dict) else None
    return (text or str(error))[:200]


def invoke_structured(schema: type[T], messages: list[BaseMessage]) -> T:
    """Ask the model for output matching `schema`, retrying transient malformed responses."""
    structured_llm = get_llm().with_structured_output(schema)
    for attempt in range(1, MAX_STRUCTURED_ATTEMPTS + 1):
        try:
            return cast(T, structured_llm.invoke(messages))
        except groq.BadRequestError as e:
            retryable = any(marker in str(e) for marker in _RETRYABLE_MARKERS)
            if not retryable or attempt == MAX_STRUCTURED_ATTEMPTS:
                raise LLMCallError(
                    f"The model returned an unusable response after {attempt} attempt(s): "
                    f"{_short_reason(e)}"
                ) from e
            logger.warning(
                f"Model returned a malformed response (attempt {attempt}/"
                f"{MAX_STRUCTURED_ATTEMPTS}), retrying: {_short_reason(e)}"
            )
    raise AssertionError("unreachable")  # loop always returns or raises
