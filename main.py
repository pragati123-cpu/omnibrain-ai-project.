from src.graph import build_graph


def main():

    graph = build_graph()

    initial_state = {
        "messages": [],
        "next": ""
    }

    result = graph.invoke(initial_state)

    print("Final state:")
    print(result)


if __name__ == "__main__":
    main()