"""
resume_parser.py
-----------------
Handles extraction of raw text from uploaded resume files (PDF, DOCX, TXT).

This module is intentionally kept separate from the UI (app.py) and from the
scoring logic (scorer.py) so any teammate can improve parsing (e.g. add OCR
for scanned PDFs, or better section detection) without touching the rest of
the app.
"""

from __future__ import annotations
import io
import re
from typing import Optional

import PyPDF2
import docx


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file given as bytes."""
    text_chunks = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    except Exception as e:
        return f"[ERROR extracting PDF text: {e}]"
    return "\n".join(text_chunks)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract raw text from a DOCX file given as bytes."""
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs]
        # Also pull text out of any tables (skills tables, etc.)
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    paragraphs.append(cell.text)
        return "\n".join(paragraphs)
    except Exception as e:
        return f"[ERROR extracting DOCX text: {e}]"


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain .txt file given as bytes."""
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[ERROR extracting TXT text: {e}]"


def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Dispatch to the correct extractor based on file extension.

    Args:
        filename: original filename (used only to detect extension)
        file_bytes: raw bytes of the uploaded file

    Returns:
        Extracted plain text (best-effort). Returns an "[ERROR ...]" string
        rather than raising, so the UI can display a per-file warning instead
        of crashing the whole batch.
    """
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower_name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif lower_name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        return "[ERROR: unsupported file type. Please upload PDF, DOCX, or TXT.]"


def clean_text(text: str) -> str:
    """Basic normalisation: collapse whitespace, strip odd control chars."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3}[-.\s]?\d{3,4}", text)
    return match.group(0) if match else None
