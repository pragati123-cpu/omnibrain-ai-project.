from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.supervisor import supervisor_node


def build_graph():

    graph = StateGraph(AgentState)

    # Add supervisor node
    graph.add_node("supervisor", supervisor_node)

    # START → supervisor
    graph.add_edge(START, "supervisor")

    # supervisor → END
    graph.add_edge("supervisor", END)

    return graph.compile()