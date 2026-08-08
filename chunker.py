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
            continue  

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
