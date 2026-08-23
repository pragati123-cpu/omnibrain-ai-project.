# OmniBrain — Week 2: Sub-Agents (Specialized Workers)

Three independent, callable agents. Each takes a natural-language
query and returns a dict with a consistent shape:
`{"agent", "query", "answer"/"results", "citations"}` — so whoever
wires the Supervisor/LangGraph state machine can call all three the
same way regardless of what's happening inside each one.

## Files

```
agents/
├── config.py         # shared Qdrant connection + path config
├── search_agent.py   # text retrieval (Qdrant "omnibrain_text")
├── vision_agent.py   # image retrieval + local LLaVA reading (Qdrant "omnibrain_images")
├── sql_agent.py       # Text-to-SQL over a sample stock price DB (still uses GPT-4o)
└── README.md
```

## Setup

```bash
pip install qdrant-client sentence-transformers torch pillow openai ollama
pip install git+https://github.com/openai/CLIP.git   # NOT `pip install clip` -- see note below
```

**Vision Agent now runs LLaVA locally via Ollama** (switched from
GPT-4o for cost/openness — SQL Agent still uses GPT-4o for Text-to-SQL):

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull the LLaVA model once:
ollama pull llava
# 3. Ollama runs as a background service after install --
#    vision_agent.py connects to it automatically at localhost:11434
```

**Environment variables required:**
```bash
export OPENAI_API_KEY=sk-...          # required for sql_agent only now
export QDRANT_HOST=localhost           # optional, defaults to localhost
export QDRANT_PORT=6333                # optional, defaults to 6333
export LLAVA_MODEL_NAME=llava          # optional, defaults to "llava"
export OLLAMA_HOST=http://localhost:11434   # optional, Ollama's default
```

**Prerequisites:**
- A running Qdrant server (`docker run -p 6333:6333 qdrant/qdrant`) with
  `omnibrain_text` and `omnibrain_images` already populated — these come
  from the teammate's `vector_store/` ingestion scripts, not from this module.
- Ollama installed, running, and `llava` pulled (see above) — for `vision_agent.py`.
- `sql_agent.py` builds its own sample database automatically on first
  run — no external setup needed for that one.

## Run each agent standalone

```bash
cd agents
python search_agent.py "What was the revenue in 2025?"
python vision_agent.py "What does the revenue chart show?"
python sql_agent.py "What was the average closing price in Q3 2025?"
```

## What's been tested vs. what needs your environment

This was built and verified in a sandbox with no live Qdrant server,
no internet access to Hugging Face or OpenAI, and no API key — so
testing was split accordingly:

**Fully tested, working correctly:**
- `sql_agent.build_sample_db()` — generates 261 rows of realistic
  synthetic daily stock data for fiscal year 2025 (ticker `ACME`),
  verified by direct SQL query against the resulting database.
- `sql_agent._is_safe_select()` — the SQL safety guardrail, tested
  against 7 cases including legitimate SELECTs, DROP/DELETE/UPDATE
  attempts, and multi-statement injection via semicolon. All passed.
- The full SQL Agent pipeline (question → generated SQL → execution →
  cited response) — tested end-to-end with the LLM call mocked to
  return a realistic SQL translation, confirming the query→execution
  →response path works correctly.
- All four files compile with no syntax errors.

**Needs your environment to verify (couldn't be tested here):**
- `search_agent.py` and `vision_agent.py`'s actual Qdrant queries —
  need a running Qdrant server with real embedded data.
- The real LLaVA calls in `vision_agent.py` — confirmed the code
  correctly reaches Ollama's connection layer (fails only with
  "Failed to connect to Ollama" since no server is running in this
  sandbox); needs Ollama installed + running + `llava` pulled to
  verify actual image reading quality.
- The real GPT-4o calls in `sql_agent.py` — need `OPENAI_API_KEY` and
  network access to api.openai.com.
- CLIP model loading in `vision_agent.py` — needs to download model
  weights on first run (one-time, needs internet).

Run the standalone commands above once your Qdrant server, Ollama,
and `OPENAI_API_KEY` are all set up — that'll be the real end-to-end check.

**A note on accuracy**: LLaVA's numeric/OCR-style reading of charts
and tables is generally less reliable than GPT-4o's. Worth testing
`vision_agent.py` against a chart/table image with numbers you
already know, to get a feel for how much to trust its output before
relying on it for a demo — this is a model-capability trade-off, not
something to try to "fix" in the code.

## Design notes

- **Search Agent has no LLM call** — it's retrieval only. Whether/how
  its results get synthesized into a final answer is the Supervisor's
  job, not this agent's.
- **Vision Agent uses local LLaVA via Ollama**, not GPT-4o — open
  source, no per-call cost, no API key needed for this agent. Trade-off:
  generally less accurate at reading precise numbers off charts/tables
  than GPT-4o. If numbers come out wrong in testing, that's most likely
  the model's limitation, not a bug — worth flagging in your writeup
  as a known trade-off of the open-source choice, not silently
  "fixing" by tweaking the prompt indefinitely.
- **CLIP package**: `vision_agent.py` uses OpenAI's official `clip`
  package (`pip install git+https://github.com/openai/CLIP.git`),
  matching what the teammate's `embed_images.py` already uses for
  consistency — NOT the unrelated `clip` package that exists on PyPI
  under the same import name. (Note: CLIP is still used here purely
  for image *retrieval* — finding the right image via embedding
  similarity — while LLaVA is only used for *reading* the chosen
  image. Different jobs, both needed.)
- **SQL Agent's sample data is clearly synthetic** (`historical_stock_prices.db`,
  seeded with `random.seed(42)` for reproducibility) — swap
  `build_sample_db()` for a real market-data loader whenever the team
  has a real source; nothing else in the agent needs to change since
  the rest only depends on the table schema.
- **SQL safety guardrail**: only single, read-only `SELECT` statements
  are executed. Anything else (DDL, writes, or a second statement via
  `;`) is rejected before it ever reaches the database.

## Known open questions for the team

- Point-ID collisions in Qdrant (flagged earlier in the ingestion/
  vector_store review) will also affect these agents if a second
  document gets ingested — worth resolving before these get tested
  against more than one PDF.
- Whoever builds the Supervisor needs to decide the final answer
  synthesis step: does it call an LLM once more over all agent
  outputs, or does one agent's raw output get returned directly?
  Not this module's job, but worth settling before integration.
