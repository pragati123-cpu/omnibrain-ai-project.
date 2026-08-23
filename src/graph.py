from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.supervisor import supervisor_node


def worker_node(state: AgentState):
    """
    Placeholder worker node.

    This will later be replaced by actual agent nodes.
    """

    print("Worker is executing the task...")

    return {
        "current_agent": "worker",
        "result": "Worker completed the task.",
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
    # Worker → Supervisor
    # -------------------------

    graph.add_edge("worker", "supervisor")

    return graph.compile()