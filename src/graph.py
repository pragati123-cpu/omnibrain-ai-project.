from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.supervisor import supervisor_node
from src.relevance_validator import validate_relevance_node
from src.observability import traced
from agents.search_agent import search_agent
from agents.vision_agent import vision_agent


@traced("graph.worker_node", as_type="span")
def worker_node(state: AgentState):
    """
    Retrieval step: calls the Search Agent against Qdrant for the
    current task/query.

    Note: search_agent() itself is already traced internally (see
    agents/search_agent.py), so this wraps the NODE as a whole --
    giving a latency figure for "how long did the graph spend in this
    node" as distinct from "how long did the underlying agent call
    take", which matters if node-level overhead (state handling,
    LangGraph's own bookkeeping) ever becomes worth measuring
    separately from the agent call itself.
    """
    query = state.get("task", "")
    print(f"Worker retrieving for query: {query!r}")

    search_result = search_agent(query)
    retrieved_docs = search_result.get("results", [])

    # Week 4 - Task 4
    # Retrieve relevant chart/image and its source information
    vision_result = vision_agent(query, top_k=1)
    visual_sources = vision_result.get("citations", [])

    return {
    "current_agent": "worker",
    "retrieved_docs": retrieved_docs,
    "visual_sources": visual_sources,
    "result": f"Retrieved {len(retrieved_docs)} chunk(s).",
    "step_count": state.get("step_count", 0) + 1,
}


def route_from_supervisor(state: AgentState):
    """
    Determines which node should execute next.
    """

    return state.get("next", "END")


def build_graph():

    graph = StateGraph(AgentState)

    # -------------------------
    # Add nodes
    # -------------------------

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("worker", worker_node)
    graph.add_node("validate_relevance", validate_relevance_node)

    # -------------------------
    # START → Supervisor
    # -------------------------

    graph.add_edge(START, "supervisor")

    # -------------------------
    # Supervisor routing
    # -------------------------

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "worker": "worker",
            "END": END,
        },
    )

    # -------------------------
    # Worker → Validate Relevance → Supervisor
    # -------------------------

    graph.add_edge("worker", "validate_relevance")
    graph.add_edge("validate_relevance", "supervisor")

    return graph.compile()
