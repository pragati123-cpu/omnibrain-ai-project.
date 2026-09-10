from langgraph.graph import StateGraph, START, END

from .state import QueryRewriteState
from .nodes import (
    search,
    evaluate_retrieval,
    rewrite_query,
    success,
    failure,
)


def build_graph():

    graph = StateGraph(QueryRewriteState)

    # Add nodes
    graph.add_node("search", search)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("success", success)
    graph.add_node("failure", failure)

    # Start
    graph.add_edge(START, "search")

    # Search → evaluate → decision
    graph.add_conditional_edges(
        "search",
        evaluate_retrieval,
        {
            "success": "success",
            "rewrite": "rewrite",
            "failure": "failure",
        },
    )

    # Rewrite → Search
    graph.add_edge("rewrite", "search")

    # End
    graph.add_edge("success", END)
    graph.add_edge("failure", END)

    return graph.compile()