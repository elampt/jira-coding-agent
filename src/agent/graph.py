"""
LangGraph agent — wires all nodes into a state machine.

Phase 6 flow:
  START → parse → search → plan → (risk check)
                                    │
                                    ├── auto-approved risk (config) → write → test → END
                                    │                        └── fail → fix → write → test (self-heal, max 3)
                                    │
                                    └── high → wait_approval → (response check)
                                                                    │
                                                                    ├── approved → write → test → END
                                                                    └── rejected → END
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes.approver import wait_for_approval
from src.agent.nodes.fixer import fix_test_failure
from src.agent.nodes.parser import parse_ticket
from src.agent.nodes.planner import plan_changes
from src.agent.nodes.searcher import search_codebase
from src.agent.nodes.tester import run_tests
from src.agent.nodes.writer import apply_changes
from src.agent.state import AgentState
from src.config import config

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def should_wait_for_approval(state: AgentState) -> str:
    """Conditional edge after PLAN node.

    Only risk levels listed in config.yaml's jira.auto_approve_risk_levels proceed without a
    human. Anything else — including an unrecognised value — pauses for approval, so a typo or
    a new risk level from the LLM fails safe.
    """
    risk = str(state["ticket_plan"].get("risk_level", "low")).strip().lower()
    if risk in {level.lower() for level in config.jira.auto_approve_risk_levels}:
        logger.info(f"{risk.capitalize()}-risk change — auto-approved by config, proceeding")
        return "write"
    logger.info(f"{risk.capitalize()}-risk change — routing to approval")
    return "wait_approval"


def should_proceed_after_approval(state: AgentState) -> str:
    """Conditional edge after WAIT_APPROVAL node.

    Routes based on human's response.
    """
    status = state.get("approval_status", "rejected")
    if status == "approved":
        logger.info("Human approved — proceeding")
        return "write"
    logger.info("Human rejected — stopping")
    return "end"


def should_retry_or_end(state: AgentState) -> str:
    """Conditional edge after TEST node.

    - Tests passed → END (success!)
    - Tests couldn't run (environment problem) → END — the fixer would only invent edits
    - Tests failed + retries left → fix
    - Tests failed + no retries left → END (give up)
    """
    if state.get("test_passed", False):
        logger.info("Tests passed — done!")
        return "end"

    if state.get("environment_failure", False):
        logger.error("Tests could not run (environment problem) — skipping fixer")
        return "end"

    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        logger.warning(f"Tests still failing after {MAX_RETRIES} attempts — giving up")
        return "end"

    logger.info(f"Tests failed — attempting fix ({retry_count + 1}/{MAX_RETRIES})")
    return "fix"


def build_agent():
    """Build and compile the LangGraph agent.

    Uses MemorySaver for checkpointing — required for interrupt/resume.
    State is saved in memory, keyed by thread_id (we use issue_key).
    """

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse", parse_ticket)
    graph.add_node("search", search_codebase)
    graph.add_node("plan", plan_changes)
    graph.add_node("wait_approval", wait_for_approval)
    graph.add_node("write", apply_changes)
    graph.add_node("test", run_tests)
    graph.add_node("fix", fix_test_failure)

    # Define the flow
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "search")
    graph.add_edge("search", "plan")

    # After PLAN: check risk level → maybe pause for approval
    graph.add_conditional_edges(
        "plan",
        should_wait_for_approval,
        {
            "write": "write",
            "wait_approval": "wait_approval",
        },
    )

    # After WAIT_APPROVAL: check response → proceed or cancel
    graph.add_conditional_edges(
        "wait_approval",
        should_proceed_after_approval,
        {
            "write": "write",
            "end": END,
        },
    )

    # WRITE → TEST (same as before)
    graph.add_edge("write", "test")

    # TEST → retry or end
    graph.add_conditional_edges(
        "test",
        should_retry_or_end,
        {
            "end": END,
            "fix": "fix",
        },
    )

    # Self-heal loop: fix → write → test
    graph.add_edge("fix", "write")

    # Checkpointer — required for interrupt/resume
    # MemorySaver stores state in RAM (lost on restart — fine for MVP)
    checkpointer = MemorySaver()

    agent = graph.compile(checkpointer=checkpointer)
    logger.info("Agent compiled: parse → search → plan → [approval?] → write → test")

    return agent


# Create the agent once — reused for every ticket
agent = build_agent()
