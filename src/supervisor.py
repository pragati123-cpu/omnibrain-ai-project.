from src.state import AgentState


def supervisor_node(state: AgentState):
    print("Supervisor is deciding what to do...")

    task = state.get("task", "")
    step_count = state.get("step_count", 0)

    if not task:
        return {
            "next": "END",
            "current_agent": "supervisor",
            "result": "No task provided.",
            "step_count": step_count,
        }

    # Stop after a few iterations for now.
    if step_count >= 2:
        return {
            "next": "END",
            "current_agent": "supervisor",
            "result": "Workflow completed.",
            "step_count": step_count,
        }

    return {
        "next": "worker",
        "current_agent": "supervisor",
        "step_count": step_count,
    }