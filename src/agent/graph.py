"""
LangGraph agent — wires all nodes into a state machine.

Phase 4 flow:
  START → parse → search → plan → write → test → END
                                            │
                                            ├── pass → END
                                            └── fail → fix → write → test (retry, max 3)
                                                        └── 3 fails → END (give up)

The test → fix → write → test loop is the "self-heal" cycle.
"""

import logging
from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes.parser import parse_ticket
from src.agent.nodes.searcher import search_codebase
from src.agent.nodes.planner import plan_changes
from src.agent.nodes.writer import apply_changes
from src.agent.nodes.tester import run_tests
from src.agent.nodes.fixer import fix_test_failure

logger = logging.getLogger(__name__)

# Maximum number of fix attempts before giving up
MAX_RETRIES = 3


def should_retry_or_end(state: AgentState) -> str:
    """Conditional edge after TEST node.

    Decides where to go next based on test results:
      - Tests passed → go to END (success!)
      - Tests failed + retries left → go to "fix" (try again)
      - Tests failed + no retries left → go to END (give up)
    """
    if state.get("test_passed", False):
        logger.info("Tests passed — done!")
        return "end"

    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        logger.warning(f"Tests still failing after {MAX_RETRIES} attempts — giving up")
        return "end"

    logger.info(f"Tests failed — attempting fix ({retry_count + 1}/{MAX_RETRIES})")
    return "fix"


def build_agent():
    """Build and compile the LangGraph agent.

    Returns a compiled graph that can be invoked with:
        result = agent.invoke({"issue_key": "KAN-5", "summary": "...", ...})
    """

    # 1. Create the graph
    graph = StateGraph(AgentState)

    # 2. Add nodes
    graph.add_node("parse", parse_ticket)
    graph.add_node("search", search_codebase)
    graph.add_node("plan", plan_changes)
    graph.add_node("write", apply_changes)
    graph.add_node("test", run_tests)
    graph.add_node("fix", fix_test_failure)

    # 3. Define the flow
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "search")
    graph.add_edge("search", "plan")
    graph.add_edge("plan", "write")
    graph.add_edge("write", "test")       # After writing → run tests

    # Conditional edge: after test, decide what to do
    graph.add_conditional_edges(
        "test",                           # From: test node
        should_retry_or_end,              # Decision function
        {
            "end": END,                   # If tests pass or max retries → done
            "fix": "fix",                 # If tests fail → go to fix node
        }
    )

    # After fix → write the fixes → test again (the loop)
    graph.add_edge("fix", "write")

    # 4. Compile
    agent = graph.compile()
    logger.info("Agent compiled: parse → search → plan → write → test (with self-heal loop)")

    return agent


# Create the agent once — reused for every ticket
agent = build_agent()
