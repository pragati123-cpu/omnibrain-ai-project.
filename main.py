from src.query_rewriter.graph import build_graph


def main():

    graph = build_graph()

    initial_state = {
        "original_query": "What is LangGraph?",
        "current_query": "What is LangGraph?",

        "results": [],
        "retrieval_score": 0.0,

        "rewritten_query": "",

        "retry_count": 0,
        "max_retries": 2,

        "status": "",
    }

    result = graph.invoke(initial_state)

    print("\n==============================")
    print("FINAL STATE")
    print("==============================")

    print(result)


if __name__ == "__main__":
    main()