from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.supervisor import supervisor_node
from src.relevance_validator import validate_relevance_node
from agents.search_agent import search_agent


def worker_node(state: AgentState):
    """
    Placeholder worker node.

    This will later be replaced by actual agent nodes.
    """

    print("Worker is executing the task...")

    return {
        "current_agent": "worker",
        "result": "Worker completed the task.",
    Retrieval step: calls the Search Agent against Qdrant for the
    current task/query.

    CHANGED from the earlier placeholder (which just returned a
    hardcoded string) to a real call into agents/search_agent.py --
    Task 1 (relevance validation) has nothing to validate without
    real retrieved data. This is a shared file; flag this change with
    the team if anyone else is also wiring real agent calls into the
    Supervisor's routing logic, to avoid duplicate/conflicting work.
    """
    query = state.get("task", "")
    print(f"Worker retrieving for query: {query!r}")

    search_result = search_agent(query)
    retrieved_docs = search_result.get("results", [])

    return {
        "current_agent": "worker",
        "retrieved_docs": retrieved_docs,
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
<<<<<<< HEAD
=======
    graph.add_node("validate_relevance", validate_relevance_node)
>>>>>>> 27f39e588985873c86b11c98d5c0d08a4f048d4d

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
<<<<<<< HEAD
    # Worker → Supervisor
    # -------------------------

    graph.add_edge("worker", "supervisor")

    return graph.compile()
=======
    # Worker → Validate Relevance → Supervisor
    #
    # ADDED validate_relevance as an unconditional stop between
    # retrieval and the Supervisor. It only ANNOTATES state
    # (is_relevant / relevance_reason / relevance_score) -- it does
    # not change routing itself. Task 2's rewriter/fallback agent is
    # what should read is_relevant and decide whether to loop back
    # for a retry; that logic doesn't exist yet and isn't added here,
    # to keep Task 1 and Task 2's responsibilities cleanly separated.
    # -------------------------

    graph.add_edge("worker", "validate_relevance")
    graph.add_edge("validate_relevance", "supervisor")

    return graph.compile()
>>>>>>> 27f39e588985873c86b11c98d5c0d08a4f048d4d
