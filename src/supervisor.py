from src.state import AgentState


def supervisor_node(state: AgentState):
    print("Supervisor is deciding what to do...")

    return {
        "next": "END"
    }