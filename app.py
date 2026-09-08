import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# 1. Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="OmniBrain Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 OmniBrain")
st.caption("AI-powered document assistant with Multi-Agent & Vision Support (Week 2 Integrated)")

# ---------------------------------------------------------
# 2. Backend Modules Import (Safety try-except blocks)
# ---------------------------------------------------------
backend_available = True
try:
    from agents.vision_agent import process_image  # Arkendu's Vision Agent
    from src.graph import build_graph               # Shuaib's LangGraph Workflow
except ImportError:
    pass


# ---------------------------------------------------------
# 3. Sidebar: Vision Support & File Upload
# ---------------------------------------------------------
st.sidebar.title("📁 Document & Vision Panel")
uploaded_file = st.sidebar.file_uploader("Upload an image for Vision Analysis...", type=["jpg", "jpeg", "png"])

vision_image = None
if uploaded_file is not None:
    vision_image = Image.open(uploaded_file)
    st.sidebar.image(vision_image, caption="Uploaded Image Preview", use_container_width=True)
    st.sidebar.success("Vision file loaded successfully!")


# ---------------------------------------------------------
# 4. Initialize Chat History State
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# 5. Chat Input & Core Execution Workflow
# ---------------------------------------------------------
if prompt := st.chat_input("Ask OmniBrain something..."):
    
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.status("OmniBrain agents are processing your request...", expanded=True) as status:
            
            assistant_response = ""
            
            try:
                # Case A: If an image is uploaded, route to Vision Agent (Arkendu's Task)
                if vision_image is not None and 'process_image' in globals():
                    st.write("Analyzing image using local LLaVA vision agent...")
                    assistant_response = process_image(vision_image, prompt)
                    # assistant_response = "Vision Agent processed the image successfully! (LLaVA integration active)."
                
                # Case B: Standard text query, route through LangGraph workflow (Shuaib's Task)
                elif 'build_graph' in globals():
                    st.write("Executing multi-agent LangGraph workflow...")
                    graph = build_graph()
                    result = graph.invoke({"messages": [prompt]})
                    assistant_response = str(result)
                    

                    # assistant_response = "LangGraph state machine executed successfully. (Backend connected)."
                
                else:
                    # Fallback if imports are missing
                    assistant_response = f"Echo Response: Received your query -> '{prompt}'. (Backend modules pending final wiring)."
                
                status.update(label="Workflow completed successfully!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="Error during execution", state="error", expanded=True)
                assistant_response = f"An error occurred while running the agents: {str(e)}"

        # Display final output
        st.markdown(assistant_response)
        
        # Store assistant response in history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})