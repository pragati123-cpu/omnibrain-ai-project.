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
# OmniBrain

OmniBrain is a document intelligence and retrieval project designed to ingest PDFs, extract usable content, embed it into a vector database, and answer user queries through a retrieval-augmented workflow.

The project currently combines:

- PDF ingestion and chunking
- vector search with Qdrant
- search agent retrieval over text embeddings
- LangGraph workflow orchestration
- a Self-RAG relevance validation step before downstream reasoning
- a Streamlit chat frontend for interaction

---

## Current project state

The repository is in a hybrid stage: the core retrieval and graph orchestration pieces are implemented, and the system is structured around a retrieval-validation flow before a final answer is generated.

### Implemented features

- PDF ingestion and chunk extraction pipeline
- Qdrant-based embeddings and search setup
- text retrieval agent using sentence-transformers
- LangGraph-based orchestration flow
- retrieval relevance validator for Self-RAG behavior
- shared state model for graph nodes
- Streamlit frontend shell for chat interaction

### In progress / not yet fully validated end-to-end

- Qdrant must be running for live retrieval testing
- downstream rewrite/retry logic for irrelevant retrievals is not yet fully implemented
- end-to-end LLM answer generation from retrieved evidence is still being integrated
- production calibration of relevance thresholds against real document data is still pending

---

## Architecture overview

```text
User Query
   │
   ▼
Frontend (Streamlit)
   │
   ▼
Supervisor / LangGraph workflow
   │
   ├── Search Agent
   │      └── Qdrant vector retrieval
   │
   └── Relevance Validator
          ├── score-based threshold check
          └── optional LLM judge path
```

The graph is defined in `src/graph.py` and the shared state is defined in `src/state.py`.

---

## Key modules

### `agents/search_agent.py`

Retrieves the top matching chunks for a query using:

- `SentenceTransformer`
- Qdrant client
- the configured text collection

It returns:

- `chunk_id`
- `text`
- `page_number`
- `source_file`
- `score`

### `src/relevance_validator.py`

This module implements the Self-RAG Retrieval Validation Logic.

Its purpose is to check whether the top retrieved chunks are actually relevant to the user query before the workflow continues to any answer-generation or retry logic.

Two validation paths are included:

1. Threshold-based validation
   - evaluates the top Qdrant similarity score
   - fast and does not require an API call

2. LLM-judge validation
   - asks the LLM to judge whether the retrieved text answers the question
   - better semantic judgment, but costs latency and model usage

The output is stored in state as:

- `is_relevant`
- `relevance_reason`
- `relevance_score`

### `src/graph.py`

Builds the LangGraph workflow with:

- `supervisor`
- `worker`
- `validate_relevance`

The retrieval node calls the search agent, then the relevance validator annotates the state before control returns to the supervisor.

### `src/state.py`

Defines the shared state shape used across the graph, including retrived chunks and relevance metadata.

### `frontend/app.py`

Provides the initial Streamlit chat interface. It is structured for future integration with the retrieval and LangGraph backend.

---

## Self-RAG retrieval validation logic

The Self-RAG step is designed to answer the critical question:

> “Did the retrieval actually return content relevant to the query, or did the vector DB return a weak or unrelated match?”

The validator checks the top retrieved result and returns a boolean `is_relevant` plus a reason string and score. This lets downstream logic decide whether to:

- answer normally,
- rewrite the query,
- retry retrieval,
- or stop with a fallback response.

Threshold logic currently uses a starting point of:

```python
RELEVANCE_SCORE_THRESHOLD = 0.35
```

This is a practical starting value for the current embedding model and should be tuned with real-world retrieval results from the project documents.

---

## Project structure

```text
omnibrain_final/
├── agents/
│   ├── config.py
│   ├── search_agent.py
│   ├── sql_agent.py
│   └── vision_agent.py
├── data/
│   ├── manifest.json
│   ├── extracted_images/
│   ├── raw_pdfs/
│   └── stock_data/
├── frontend/
│   └── app.py
├── ingestion/
│   ├── chunker.py
│   ├── pdf_parser.py
│   ├── pipeline.py
│   ├── README.md
│   └── table_extractor.py
├── src/
│   ├── graph.py
│   ├── relevance_validator.py
│   ├── state.py
│   ├── supervisor.py
│   └── ...
├── vector_store/
│   ├── check_qdrant.py
│   ├── embed_images.py
│   ├── embed_text.py
│   ├── ingest_to_qdrant.py
│   └── qdrant_setup.py
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
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
## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
. .venv/Scripts/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Run ingestion or retrieval workflows

Examples:

```bash
python vector_store/ingest_to_qdrant.py
python agents/search_agent.py "What was the revenue in 2025?"
```

### 5. Run the frontend

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
---

## Current limitations

The project is functional as a structured retrieval pipeline, but a few items still need deeper integration work:

- Qdrant must be running for retrieval to work in real time
- relevance threshold tuning still needs actual document-level calibration
- fallback/rewrite loop for irrelevant retrievals is not yet fully implemented
- final response synthesis from retrieved context is not yet fully completed in the graph
- some files still reflect earlier milestones or demo-oriented assumptions

---

## Project status

**Overall status: Code foundation implemented; end-to-end retrieval and workflow validation is partially complete but still requires live environment verification.**

This README reflects the current repository state, including the implemented Self-RAG retrieval validation logic and the LangGraph-based retrieval workflow.
## Week 4 - Chart & Visual Source Mapping

Implemented chart and visual source mapping for the OmniBrain project.

### Features
- Retrieves relevant charts and visual content using the Vision Agent.
- Maps retrieved visuals to their original PDF source.
- Displays the source PDF file name.
- Displays the exact PDF page number.
- Identifies the visual type such as chart, figure, or table.
- Provides visual source information along with the AI response.

### Example
AI Answer:
The revenue chart shows the revenue trend mentioned in the provided document.

📊 Referenced Chart / Figure

📄 Source: STXIntimationUFRs24072025signed.pdf  
📍 Page: 12  
🔎 Type: embedded_figure
