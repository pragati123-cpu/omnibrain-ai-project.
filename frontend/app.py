import streamlit as st


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="OmniBrain Chat",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Header
# -----------------------------
st.title("🤖 OmniBrain")
st.caption("AI-powered document assistant")


# -----------------------------
# Initialize Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Render Chat History
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat Input
# -----------------------------
user_prompt = st.chat_input(
    "Ask OmniBrain something..."
)


# -----------------------------
# Handle User Message
# -----------------------------
if user_prompt:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    # Render user message
    with st.chat_message("user"):
        st.markdown(user_prompt)


    # Temporary assistant response
    assistant_response = (
        "This is a demo response. "
        "The OmniBrain backend will be connected later."
    )


    # Store assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response
        }
    )


    # Render assistant response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)