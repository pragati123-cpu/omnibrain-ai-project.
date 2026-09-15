import streamlit as st

st.set_page_config(page_title="OmniBrain UI", page_icon="🧠", layout="wide")

st.title("🧠 OmniBrain: Agentic Multi-Modal RAG")
st.write("Welcome to the chat interface. Ask questions about your financial documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        thought_process = "Thinking: Analyzing the query and fetching data from vector database..."
        response_text = f"Here is the answer based on your query: '{user_query}'"
        
        with st.expander("Agent's Thought Process"):
            st.write(thought_process)
            
        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
