# AI-Based Resume Screening Application — Frontend (Streamlit)

This is **Task 5: Frontend UI & State Integration** for the Micro Project
"Resume Based Screening Application".

## What this covers
- A Streamlit UI: job description input, multi-resume upload (PDF/DOCX/TXT),
  ranked results table, per-candidate detail view, CSV export.
- `st.session_state` used throughout so uploaded resumes, the job
  description, and computed results survive Streamlit reruns (button
  clicks, filter changes, etc.) without being lost.
- Parsing and scoring logic kept in separate modules (`utils/`) so the
  UI layer doesn't need to change if a teammate swaps in a better model.

## Project structure
```
resume_screening_app/
├── app.py                  # Streamlit UI + session state (this task)
├── utils/
│   ├── __init__.py
│   ├── resume_parser.py    # PDF/DOCX/TXT text extraction
│   └── scorer.py           # TF-IDF + cosine similarity matching
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## How it works
1. Paste a job description in the sidebar.
2. Upload one or more resumes (PDF, DOCX, or TXT).
3. Click **Analyze Resumes**.
4. View ranked candidates by match score, with matched/missing keywords,
   auto-detected email/phone, and a text preview per resume.
5. Download the ranked results as a CSV.

## Where to extend for other tasks
- **Better matching model**: replace the body of `score_resumes()` in
  `utils/scorer.py` (e.g. swap TF-IDF for sentence embeddings or a trained
  classifier) — the function signature can stay the same, so `app.py`
  needs no changes.
- **Better parsing**: extend `utils/resume_parser.py` (e.g. OCR for scanned
  PDFs, structured section extraction for "Skills"/"Experience").
- **Persistence/DB**: if the project needs to save results permanently,
  add a `utils/storage.py` and call it from `app.py` after scoring —
  the session-state pattern here can be swapped for a database read/write.
