# OmniBrain - Multi-Modal & Agentic AI Assistant (Frontend Module)
> **Team Lead:** Pragati Shukla | **Team Members:** Shuaib, Keshav, Arkendu, Pramod

---
# Task 4 – Chat UI Layout & Rendering


## Overview

This task implements a clean and interactive chat user interface for the **OmniBrain** project using **Streamlit**.

The purpose of this task is to create the frontend chat interface with support for user input, chat history, and message rendering. The UI is designed as a standalone frontend component and does not depend on the Week 1 implementation.

---

## Objectives

The main objectives of this task are:

* Create a new frontend structure for the OmniBrain project.
* Build a chat interface using Streamlit.
* Provide a chat input field for user queries.
* Maintain chat history using Streamlit session state.
* Render user messages and assistant responses separately.
* Create a clean and simple UI suitable for future backend integration.

---

## Project Structure

```text
omnibrain-ai-project
│
└── frontend
    └── app.py
```

---

## Technologies Used

* **Python**
* **Streamlit 1.54.0**

---

## Features

### 1. Chat Interface

The application provides a simple and user-friendly chat interface for interacting with OmniBrain.

### 2. Chat Input

Users can enter their questions or messages through the Streamlit chat input component.

```python
st.chat_input()
```

### 3. Chat History

Chat messages are stored using Streamlit's session state:

```python
st.session_state.messages
```

This allows previous messages to remain visible during the current session.

### 4. User Message Rendering

User messages are displayed using:

```python
st.chat_message("user")
```

### 5. Assistant Message Rendering

Assistant responses are displayed using:

```python
st.chat_message("assistant")
```

### 6. Demo Assistant Response

For the current task, a temporary/demo assistant response is used.

The actual OmniBrain backend and AI model will be integrated in a future implementation.

---

## Application Flow

```text
User
  │
  ▼
Chat Input
  │
  ▼
User Message
  │
  ▼
Session State
  │
  ▼
Assistant Response
  │
  ▼
Chat History
```

---

## Running the Application

### Step 1 – Install Streamlit

If Streamlit is not already installed:

```bash
pip install streamlit
```

### Step 2 – Run the Application

From the project root directory:

```bash
python -m streamlit run frontend/app.py
```

Alternatively, navigate to the frontend directory:

```bash
cd frontend
python -m streamlit run app.py
```

### Step 3 – Open the Application

After starting the application, Streamlit will provide a local URL:

```text
http://localhost:8501
```

Open this URL in a web browser.

---

## Testing

The following functionality was tested:

* Chat input accepts user messages.
* User messages are displayed correctly.
* Assistant responses are displayed correctly.
* Multiple messages remain visible in the chat history.
* Session state maintains the conversation during the active session.
* Streamlit application runs successfully on the local machine.

---

## Current Limitations

The current implementation is focused only on the frontend chat UI.

The following components are not yet connected:

* LLM/API backend
* FastAPI
* LangGraph
* Qdrant vector database
* Document retrieval
* Real AI-generated responses

The assistant response is currently a demo response.

---

## Future Integration

The frontend can later be connected to the OmniBrain backend using the following architecture:

```text
Streamlit Chat UI
       │
       ▼
    FastAPI
       │
       ▼
   LangGraph
       │
       ├──► Qdrant
       │
       └──► LLM
       │
       ▼
  AI Response
       │
       ▼
Streamlit Chat UI
```

---

## Git Branch

The implementation was developed on:

```text
Pramod-week-2
```

The Task 4 frontend file is:

```text
frontend/app.py
```

---

## Task Status

**Task 4 – Chat UI Layout & Rendering: Completed ✅**

The Streamlit-based chat interface is successfully implemented, tested locally, and prepared for future backend integration.
