"""
LangGraph agent — wires all nodes into a state machine.

Flow: START → parse → search → plan → write → END

Each node reads from AgentState, does its work, returns a partial update.
LangGraph manages the state and passes it between nodes automatically.

Phase 2: Linear flow (all tickets go through every node)
Phase 6: Will add conditional edges for human-in-the-loop
"""

import logging
from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes.parser import parse_ticket
from src.agent.nodes.searcher import search_codebase
from src.agent.nodes.planner import plan_changes
from src.agent.nodes.writer import apply_changes

logger = logging.getLogger(__name__)


def build_agent():
    """Build and compile the LangGraph agent.

    Returns a compiled graph that can be invoked with:
        result = agent.invoke({"issue_key": "KAN-5", "summary": "...", ...})
    """

    # 1. Create the graph with our state type
    graph = StateGraph(AgentState)

    # 2. Add nodes — each is a function that takes state, returns partial update
    graph.add_node("parse", parse_ticket)
    graph.add_node("search", search_codebase)
    graph.add_node("plan", plan_changes)
    graph.add_node("write", apply_changes)

    # 3. Define the flow: START → parse → search → plan → write → END
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "search")
    graph.add_edge("search", "plan")
    graph.add_edge("plan", "write")
    graph.add_edge("write", END)

    # 4. Compile into a runnable agent
    agent = graph.compile()
    logger.info("Agent compiled: parse → search → plan → write")

    return agent


# Create the agent once — reused for every ticket
agent = build_agent()
