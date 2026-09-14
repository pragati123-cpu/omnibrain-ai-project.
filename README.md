# OmniBrain

OmniBrain is a multi-modal document intelligence application. It ingests PDF content, stores text and image embeddings in Qdrant, and routes user questions through retrieval agents and a LangGraph workflow.

## Current Status

Implemented:

- PDF parsing, chunking, table extraction, and image extraction
- Text and image embedding pipelines for Qdrant
- Text retrieval, SQL, and vision agents
- LangGraph orchestration with Self-RAG relevance validation
- Streamlit chat application with NeMo Guardrails
- Langfuse tracing and local backend latency metrics

Not fully validated end to end:

- Live Qdrant retrieval requires a running Qdrant instance and populated collections
- Vision queries require Ollama with the configured LLaVA model
- SQL queries require `OPENAI_API_KEY`
- Langfuse dashboard tracing requires Langfuse credentials
- Query rewrite/retry and final answer synthesis still require further integration

## Architecture

```text
User Query
    |
    v
Streamlit root app (app.py)
    |
    +--> NeMo Guardrails
    |
    +--> LangGraph workflow
           |
           +--> Search Agent --> SentenceTransformer --> Qdrant
           |
           +--> Relevance Validator
           |
           +--> Vision / SQL agent paths
```

The main application is `app.py`. The standalone UI demonstration is `frontend/app.py` and returns a placeholder response without connecting to the backend.

## Latency Tracking and Observability

`src/observability.py` provides the `@traced` decorator used across the query pipeline and agents.

Each traced operation records:

- execution duration
- success and error counts
- mean, minimum, maximum, p50, and p95 latency

The root Streamlit app displays local latency statistics in the sidebar. Langfuse tracing is enabled when these environment variables are configured:

```text
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

The decorator supports both synchronous and asynchronous functions. The current agent implementations are primarily synchronous; asynchronous query endpoint support is reserved for a future API layer.

## Setup

Create and activate a virtual environment from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements file pins a compatible LangChain 0.x stack because NeMo Guardrails 0.17.0 does not support the LangChain 1.x stack.

## External Services and Configuration

### Qdrant

Start Qdrant locally:

```bash
docker run --name omnibrain-qdrant -p 6333:6333 qdrant/qdrant
```

Then create/populate the collections as needed:

```bash
python vector_store/qdrant_setup.py
python vector_store/ingest_to_qdrant.py
```

### Ollama

Install Ollama and pull the configured vision model before using vision retrieval:

```bash
ollama pull llava
```

Configuration values are defined in `agents/config.py`.

### OpenAI

Set `OPENAI_API_KEY` before using the SQL agent or the optional LLM relevance judge.

## Running the Application

Run the integrated application from the repository root:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` when Streamlit starts.

To run the standalone frontend demonstration:

```bash
python -m streamlit run frontend/app.py
```

## Useful Commands

Run the graph entry point:

```bash
python main.py
```

Run standalone retrieval:

```bash
python agents/search_agent.py "What was the revenue in 2025?"
```

Run the guardrail smoke test:

```bash
python test_guardrails.py
```

Run repository tests when the test environment is installed:

```bash
python -m pytest -q
```

Compile-check the Python source without starting external services:

```bash
python -m compileall -q app.py main.py agents ingestion src vector_store
```

## Project Structure

```text
omnibrain_final/
├── agents/              # Search, SQL, vision agents, and configuration
├── data/                # Manifests, source PDFs, extracted images, and stock data
├── frontend/            # Standalone Streamlit UI demonstration
├── guardrails/          # NeMo Guardrails configuration
├── ingestion/           # PDF parsing, chunking, and table extraction
├── src/                 # Graph, state, validation, and observability
├── vector_store/        # Qdrant setup, embedding, and ingestion utilities
├── app.py               # Integrated Streamlit application
├── main.py              # Graph entry point
├── requirements.txt     # Pinned dependencies
└── test_guardrails.py   # Guardrail smoke test
```

## Validation Notes

The dependency manifest resolves successfully and the Python source compiles. Full runtime validation still depends on installing the requirements and starting the required external services. Retrieval, vision, SQL, and Langfuse behavior should be tested with those services configured before treating the application as production-ready.
