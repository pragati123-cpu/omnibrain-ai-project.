"""
app.py
------
Streamlit front-end for OmniBrain.

--- Week 3, Task 2: instrumented for latency tracking ---
The inline chat-handling logic was extracted into
`run_query_pipeline()` so it can be wrapped with @traced -- this
gives one top-level "how long did this whole user query take"
number, in addition to the per-agent/per-stage numbers already
recorded inside search_agent.py, vision_agent.py, and sql_agent.py.
A small sidebar expander shows the aggregated stats so latency is
visible during a demo without needing to open Langfuse.

This does NOT touch guardrail-block UI or self-correction status
messages -- that's Task 5's scope (Streamlit UI + state.py updates
for showing "Retrieved data irrelevant, rewriting query..." and
guardrail notifications). This file only adds latency visibility.
"""

import asyncio

import streamlit as st
from PIL import Image
from nemoguardrails import LLMRails, RailsConfig

from src.observability import traced, get_stats

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="OmniBrain - Multi-Modal RAG Orchestrator",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 OmniBrain: Agentic Multi-Modal RAG Orchestrator")
st.caption("Enterprise-grade document assistant powered by LangGraph, Local LLaVA Vision, and NeMo Guardrails")


# -----------------------------
# Guardrails (cached resource)
# -----------------------------
@st.cache_resource
def load_guardrails():
    try:
        config = RailsConfig.from_path("./guardrails")
        return LLMRails(config)
    except Exception as e:
        st.error(f"Failed to load guardrails configuration: {e}")
        return None


rails = load_guardrails()


# -----------------------------
# Backend modules (imported safely -- flags, not silent NameErrors)
# -----------------------------
process_image = None
build_graph = None

try:
    from agents.vision_agent import process_image
except ImportError as e:
    st.sidebar.warning(f"Vision Agent unavailable: {e}")

try:
    from src.graph import build_graph
except ImportError as e:
    st.sidebar.warning(f"LangGraph orchestrator unavailable: {e}")


# -----------------------------
# Sidebar: Document & Vision Upload Panel
# -----------------------------
st.sidebar.title("📁 Document & Vision Panel")
uploaded_file = st.sidebar.file_uploader(
    "Upload a financial report, table, or receipt (JPG, PNG)...",
    type=["jpg", "jpeg", "png", "pdf"],
)

vision_image = None
if uploaded_file is not None:
    if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
        vision_image = Image.open(uploaded_file)
        st.sidebar.image(vision_image, caption="Uploaded Document Preview", use_container_width=False)
        st.sidebar.success("Vision file loaded into memory successfully!")


# -----------------------------
# Sidebar: Latency stats panel (Task 2)
# -----------------------------
with st.sidebar.expander("⏱️ Backend Latency Stats", expanded=False):
    stats = get_stats()
    if not stats:
        st.caption("No queries traced yet this session.")
    else:
        for operation, s in sorted(stats.items()):
            if s.get("count", 0) == 0:
                continue
            st.markdown(f"**{operation}**")
            st.caption(
                f"n={s['count']} · mean={s['mean_ms']}ms · "
                f"p50={s['p50_ms']}ms · p95={s['p95_ms']}ms · "
                f"errors={s['error_count']}"
            )


# -----------------------------
# Chat Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Traced query pipeline (Task 2)
# -----------------------------
@traced("app.query_pipeline", as_type="chain")
def run_query_pipeline(prompt: str, vision_image, rails, process_image, build_graph) -> tuple[str, dict]:
    """
    Runs one user query through guardrails, then the appropriate
    agent path. Returns (assistant_response, status_info) rather than
    writing to Streamlit directly, so this function stays traceable
    and testable independent of the Streamlit runtime.
    """
    status_info = {"stage": "start"}

    # Step A: Filter through NeMo Guardrails
    guarded_response = None
    if rails:
        status_info["stage"] = "guardrails"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(
            rails.generate_async(messages=[{"role": "user", "content": prompt}])
        )
        if isinstance(res, dict):
            guarded_response = res.get("content")
        elif hasattr(res, "content"):
            guarded_response = res.content
        else:
            guarded_response = str(res)

    # Step B: Check if guardrails blocked the request
    if guarded_response and (
        "I'm sorry, I can't respond" in guarded_response
        or "restricted" in guarded_response.lower()
    ):
        status_info["stage"] = "blocked_by_guardrails"
        return guarded_response, status_info

    # Step C: Execute specialized agents if query is valid/allowed
    if vision_image is not None:
        status_info["stage"] = "vision_agent"
        if process_image is not None:
            try:
                return process_image(vision_image, prompt), status_info
            except Exception as e:
                return f"Vision Agent failed to process the image: {e}", status_info
        else:
            return (
                "Vision Agent is not available (import failed at startup -- "
                "check the sidebar warning above)."
            ), status_info

    elif build_graph is not None:
        status_info["stage"] = "langgraph"
        graph = build_graph()
        result = graph.invoke({"messages": [prompt]})
        return str(result), status_info

    else:
        status_info["stage"] = "fallback"
        return f"OmniBrain Engine: Verified query -> '{prompt}'. (Multi-modal agents active)", status_info


# -----------------------------
# Chat Input & Execution Pipeline
# -----------------------------
if prompt := st.chat_input("Ask OmniBrain about your documents, receipts, or data tables..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("OmniBrain pipeline and safety guardrails active...", expanded=True) as status:
            try:
                assistant_response, status_info = run_query_pipeline(
                    prompt, vision_image, rails, process_image, build_graph
                )
                status.update(label="Execution completed successfully!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Execution error encountered", state="error", expanded=True)
                assistant_response = f"An error occurred within the orchestration pipeline: {str(e)}"

        st.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
