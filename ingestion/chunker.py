"""
chunker.py
----------
Step 2 of the OmniBrain ingestion pipeline.

Splits per-page text into overlapping chunks suitable for embedding.

Why chunk at all?
  - Embedding models have limited context and lose precision on very
    long inputs (a whole page mixes multiple topics).
  - Smaller, semantically coherent chunks give more precise retrieval:
    the vector search returns the *specific* paragraph relevant to the
    query, not an entire page of mixed content.

Why RecursiveCharacterTextSplitter?
  - It tries to split on the "most natural" boundary first (paragraph
    breaks), and only falls back to sentence/word/character splits if
    a paragraph is too long. This keeps chunks readable and coherent,
    rather than cutting mid-sentence like a naive fixed-size splitter.

Overlap (chunk_overlap) matters for financial documents specifically:
  a sentence like "...revenue grew 12% year-over-year, driven primarily
  by..." often gets split mid-thought. Overlap ensures the tail of one
  chunk reappears at the head of the next, so the SQL/Search agents
  don't lose that connecting context.
"""

from dataclasses import dataclass, asdict
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    page_number: int
    source_file: str
    chunk_index_on_page: int


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    """
    pages: list of dicts like {"page_number": int, "text": str, "source_file": str}
           (this is exactly what pdf_parser.parse_pdf()["pages"] gives you)

    chunk_size / chunk_overlap are in characters here (simplest to reason
    about for week 1). You can switch to a token-based length function
    later once you've picked your embedding model's tokenizer.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # paragraph -> sentence -> word -> char
    )

    all_chunks: list[TextChunk] = []

    for page in pages:
        page_number = page["page_number"]
        source_file = page["source_file"]
        text = page["text"].strip()

        if not text:
            continue  # skip blank pages (common on section-divider pages)

        raw_chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{source_file}_p{page_number}_c{i}"
            all_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    page_number=page_number,
                    source_file=source_file,
                    chunk_index_on_page=i,
                )
            )

    return all_chunks


if __name__ == "__main__":
    # quick manual smoke test
    sample_pages = [
        {
            "page_number": 1,
            "source_file": "demo.pdf",
            "text": (
                "Total revenue for fiscal year 2025 was $4.2 billion, an increase "
                "of 12% year-over-year. This growth was driven primarily by strong "
                "performance in the cloud infrastructure segment, which grew 28% "
                "compared to the prior year.\n\n"
                "Operating margin improved to 22.5%, up from 19.8% in the prior "
                "fiscal year, reflecting cost discipline across the organization."
            ),
        }
    ]
    chunks = chunk_pages(sample_pages, chunk_size=120, chunk_overlap=20)
    for c in chunks:
        print(asdict(c))
