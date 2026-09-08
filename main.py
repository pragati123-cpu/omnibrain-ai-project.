from src.graph import build_graph


def main():

    graph = build_graph()

    initial_state = {
        "messages": [],
        "next": "",
        "current_agent": "",
        "task": "Test multi-step agent workflow",
        "result": "",
        "step_count": 0,
    }

    result = graph.invoke(initial_state)

    print("\nFinal state:")
    print(result)


if __name__ == "__main__":
    main()