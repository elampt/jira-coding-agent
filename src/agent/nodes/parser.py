"""
PARSE node — extracts structured intent from raw Jira ticket text.

Input:  Raw summary + description from Jira ticket
Output: TicketPlan with intent, component_hints, and risk_level

Uses the LLM with structured output — the LLM returns a Pydantic model
instead of free text, so we get typed, validated data automatically.
"""

import logging
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.state import AgentState
from src.config import secrets

logger = logging.getLogger(__name__)


# --- Pydantic model for structured LLM output ---

class TicketPlanOutput(BaseModel):
    """What we want the LLM to return — structured, validated."""
    intent: str = Field(description="What the ticket wants done in 5-10 words")
    component_hints: list[str] = Field(
        description="Keywords to search for in the codebase. Include: "
        "component names, text strings mentioned, CSS properties, file names"
    )
    risk_level: str = Field(
        description="low = text/CSS only, medium = component logic, high = multiple files or routing"
    )


# --- System prompt ---

SYSTEM_PROMPT = """You are a ticket parser for an AI coding agent that modifies a React frontend codebase.

Given a Jira ticket (summary + description), extract:
1. intent — what needs to change, in 5-10 words
2. component_hints — keywords to grep for in the codebase (component names, exact text strings, CSS class names, file names)
3. risk_level — "low" if only text/CSS changes, "medium" if component props/state change, "high" if multiple components or routing

Be specific with component_hints. If the ticket says "change the header text from 'Learn React' to 'My App'",
the hints should include: ["header", "Learn React", "App.js"] — exact strings that would appear in the code.

Always include the exact old text/value that needs to be found in the code."""


# --- Node function ---

def parse_ticket(state: AgentState) -> dict:
    """PARSE node — called by LangGraph.

    Reads: summary, description from state
    Writes: ticket_plan to state

    Returns a partial state update — only the fields this node changes.
    """
    summary = state["summary"]
    description = state.get("description", "")

    logger.info(f"Parsing ticket: {summary}")

    # Create the LLM with structured output
    llm = ChatGroq(
        api_key=secrets.groq_api_key,
        model="llama-3.3-70b-versatile",
    )
    structured_llm = llm.with_structured_output(TicketPlanOutput)

    # Ask the LLM to parse the ticket
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Summary: {summary}\n\nDescription: {description}"),
    ]
    result = structured_llm.invoke(messages)

    logger.info(f"Parsed: intent={result.intent}, hints={result.component_hints}, risk={result.risk_level}")

    # Return partial state update
    return {
        "ticket_plan": {
            "intent": result.intent,
            "component_hints": result.component_hints,
            "risk_level": result.risk_level,
        }
    }
