"""
scorer.py
---------
Matching / scoring logic that compares a Job Description (JD) against a set
of resume texts.

Currently uses TF-IDF + cosine similarity as a lightweight, dependency-light
"AI" baseline. This is deliberately isolated behind the `score_resumes()`
function so that whoever owns the ML/NLP part of the project (embeddings,
a trained classifier, an LLM-based matcher, etc.) can swap the internals
without the Streamlit UI needing any changes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ResumeResult:
    filename: str
    score: float          # 0-100
    matched_keywords: List[str]
    missing_keywords: List[str]
    raw_text: str


def _top_keywords_from_jd(jd_text: str, top_n: int = 20) -> List[str]:
    """Pull the top-N most informative words/phrases out of the JD using TF-IDF."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=top_n,
    )
    try:
        vectorizer.fit([jd_text])
        return list(vectorizer.get_feature_names_out())
    except ValueError:
        # JD too short / empty after stopword removal
        return []


def score_resumes(jd_text: str, resumes: dict[str, str]) -> List[ResumeResult]:
    """
    Score each resume's text against the job description.

    Args:
        jd_text: the job description text
        resumes: dict mapping filename -> extracted resume text

    Returns:
        List of ResumeResult, sorted by score descending.
    """
    if not jd_text.strip() or not resumes:
        return []

    filenames = list(resumes.keys())
    corpus = [jd_text] + [resumes[f] for f in filenames]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors).flatten()

    jd_keywords = set(_top_keywords_from_jd(jd_text))

    results: List[ResumeResult] = []
    for idx, filename in enumerate(filenames):
        resume_text_lower = resumes[filename].lower()
        matched = sorted([kw for kw in jd_keywords if kw in resume_text_lower])
        missing = sorted([kw for kw in jd_keywords if kw not in resume_text_lower])

        results.append(
            ResumeResult(
                filename=filename,
                score=round(float(similarities[idx]) * 100, 2),
                matched_keywords=matched,
                missing_keywords=missing,
                raw_text=resumes[filename],
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results
