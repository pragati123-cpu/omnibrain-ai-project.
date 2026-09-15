"""
app.py
------
Task 5: Frontend UI & State Integration (Streamlit)

Main entry point for the AI-based Resume Screening Application.
Responsibilities of this file ONLY:
  - Render the UI (job description input, resume uploader, results view)
  - Manage st.session_state so uploads/results persist across reruns
  - Call into utils/resume_parser.py and utils/scorer.py for the actual work

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from utils.resume_parser import extract_text, clean_text, extract_email, extract_phone
from utils.scorer import score_resumes, ResumeResult


# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Session state initialisation
# --------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "job_description": "",
        "uploaded_resumes": {},   # filename -> raw text
        "results": [],            # list[ResumeResult]
        "analyzed": False,
        "min_score_filter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --------------------------------------------------------------------------
# Sidebar: Job Description + controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Job Description")
    st.session_state.job_description = st.text_area(
        "Paste the job description here",
        value=st.session_state.job_description,
        height=250,
        placeholder="e.g. Looking for a Python developer with experience in "
                    "Django, REST APIs, SQL, and AWS...",
    )

    st.divider()
    st.header("2. Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload one or more resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    # Persist newly uploaded files into session_state so they survive reruns
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.uploaded_resumes:
                raw_text = extract_text(f.name, f.getvalue())
                st.session_state.uploaded_resumes[f.name] = clean_text(raw_text)

    if st.session_state.uploaded_resumes:
        st.caption(f"{len(st.session_state.uploaded_resumes)} resume(s) ready")
        with st.expander("Manage uploaded resumes"):
            for fname in list(st.session_state.uploaded_resumes.keys()):
                col1, col2 = st.columns([4, 1])
                col1.write(fname)
                if col2.button("Remove", key=f"remove_{fname}"):
                    del st.session_state.uploaded_resumes[fname]
                    st.session_state.analyzed = False
                    st.rerun()

    st.divider()
    st.header("3. Run")
    analyze_clicked = st.button("🔍 Analyze Resumes", type="primary", use_container_width=True)
    reset_clicked = st.button("♻️ Reset All", use_container_width=True)

    if reset_clicked:
        for key in ["job_description", "uploaded_resumes", "results", "analyzed"]:
            del st.session_state[key]
        init_session_state()
        st.rerun()


# --------------------------------------------------------------------------
# Trigger scoring (state transition)
# --------------------------------------------------------------------------
if analyze_clicked:
    if not st.session_state.job_description.strip():
        st.sidebar.error("Please paste a job description first.")
    elif not st.session_state.uploaded_resumes:
        st.sidebar.error("Please upload at least one resume.")
    else:
        with st.spinner("Scoring resumes against job description..."):
            st.session_state.results = score_resumes(
                st.session_state.job_description,
                st.session_state.uploaded_resumes,
            )
            st.session_state.analyzed = True


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("📄 AI-Based Resume Screening Application")
st.caption("Upload resumes, paste a job description, and get ranked candidate matches.")

if not st.session_state.analyzed:
    st.info(
        "👈 Paste a job description and upload resumes in the sidebar, "
        "then click **Analyze Resumes** to see ranked results here."
    )
else:
    results: list[ResumeResult] = st.session_state.results

    if not results:
        st.warning("No results to show. Try re-running the analysis.")
    else:
        # ---- Summary metrics ----
        col1, col2, col3 = st.columns(3)
        col1.metric("Resumes analyzed", len(results))
        col2.metric("Top score", f"{results[0].score:.1f}%")
        avg_score = sum(r.score for r in results) / len(results)
        col3.metric("Average score", f"{avg_score:.1f}%")

        st.divider()

        # ---- Filter control ----
        st.session_state.min_score_filter = st.slider(
            "Minimum match score to display (%)",
            min_value=0, max_value=100,
            value=st.session_state.min_score_filter,
        )
        filtered = [r for r in results if r.score >= st.session_state.min_score_filter]

        # ---- Results table ----
        st.subheader(f"Ranked Candidates ({len(filtered)} shown)")
        table_data = [
            {
                "Rank": i + 1,
                "Filename": r.filename,
                "Match Score (%)": r.score,
                "Matched Keywords": len(r.matched_keywords),
                "Missing Keywords": len(r.missing_keywords),
            }
            for i, r in enumerate(filtered)
        ]
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ---- Download results ----
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download results as CSV",
            data=csv,
            file_name="resume_screening_results.csv",
            mime="text/csv",
        )

        st.divider()

        # ---- Per-candidate detail ----
        st.subheader("Candidate Details")
        for i, r in enumerate(filtered):
            with st.expander(f"#{i + 1} — {r.filename} — {r.score:.1f}% match"):
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.markdown("**Contact info (auto-detected)**")
                    email = extract_email(r.raw_text)
                    phone = extract_phone(r.raw_text)
                    st.write(f"📧 Email: {email or 'Not found'}")
                    st.write(f"📞 Phone: {phone or 'Not found'}")

                    st.markdown("**Matched keywords**")
                    st.write(", ".join(r.matched_keywords) if r.matched_keywords else "None")

                with dcol2:
                    st.markdown("**Missing keywords (in JD, not in resume)**")
                    st.write(", ".join(r.missing_keywords) if r.missing_keywords else "None")

                st.markdown("**Extracted resume text (preview)**")
                st.text_area(
                    label="",
                    value=r.raw_text[:2000] + ("..." if len(r.raw_text) > 2000 else ""),
                    height=150,
                    key=f"preview_{r.filename}",
                    disabled=True,
                )
