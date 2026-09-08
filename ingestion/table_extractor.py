"""
table_extractor.py
-------------------
Rasterizes detected table regions into IMAGES (not structured SQL
data) -- this matches the documented OmniBrain architecture:

    "It parses tables using a Vision-Language Model (VLM)..." and
    Week 1's own task says to embed "both modalities" (text + image)
    into Qdrant -- there is no third structured/SQL modality for
    PDF tables in the spec. The Text-to-SQL agent is scoped to
    external "historical stock data", not tables inside the document.

So tables are treated the same way as charts/graphs: rasterize the
table's region of the page into a PNG, let the image-embedding step
(your teammate's part) embed it via CLIP into the same image
collection as other figures, and let the VLM read the actual numbers
out of it at query time (this is what "Vision Check" in the roadmap
describes).

pdfplumber is still used to *detect* where tables are (find_tables),
but instead of extracting cell text into rows/records, we hand its
bounding box to PyMuPDF to render that exact region of the page as a
high-resolution image.

We still exclude the table's text from the prose extraction step
below -- not because it becomes structured data, but because the
table is now represented as an image; leaving its text also in a
prose chunk would just duplicate/garble the same numbers.
"""

import os
import pdfplumber
import fitz  # PyMuPDF
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TableImage:
    image_id: str
    page_number: int          # 1-indexed
    source_file: str
    file_path: str
    width: int
    height: int
    bbox: tuple                 # (x0, top, x1, bottom) in PDF points, for reference


def extract_table_regions_as_images(
    pdf_path: str,
    output_dir: str,
    zoom: float = 3.0,          # render at 3x resolution for VLM readability
    padding: float = 4.0,       # small padding around the detected table bbox
) -> list[TableImage]:
    os.makedirs(output_dir, exist_ok=True)
    source_file = Path(pdf_path).name
    doc_name = Path(source_file).stem

    doc = fitz.open(pdf_path)
    table_images = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, plumber_page in enumerate(pdf.pages):
            page_number = page_index + 1
            found_tables = plumber_page.find_tables()
            if not found_tables:
                continue

            fitz_page = doc[page_index]

            for t_index, table in enumerate(found_tables):
                x0, top, x1, bottom = table.bbox
                # pad a little so borders/labels aren't clipped
                rect = fitz.Rect(
                    max(0, x0 - padding),
                    max(0, top - padding),
                    x1 + padding,
                    bottom + padding,
                )

                mat = fitz.Matrix(zoom, zoom)
                pix = fitz_page.get_pixmap(matrix=mat, clip=rect)

                image_id = f"{doc_name}_p{page_number}_table{t_index}"
                file_path = os.path.join(output_dir, f"{image_id}.png")
                pix.save(file_path)

                table_images.append(
                    TableImage(
                        image_id=image_id,
                        page_number=page_number,
                        source_file=source_file,
                        file_path=file_path,
                        width=pix.width,
                        height=pix.height,
                        bbox=tuple(table.bbox),
                    )
                )

    doc.close()
    return table_images


def extract_text_excluding_tables(pdf_path: str) -> list[dict]:
    """
    Extracts page text the same way the rest of the pipeline expects
    (list of {page_number, text, source_file}), but skips any words
    that fall inside a detected table's bounding box -- so table
    numbers don't ALSO end up duplicated inside a prose text chunk.
    """
    source_file = Path(pdf_path).name
    pages_out = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            table_bboxes = [t.bbox for t in page.find_tables()]

            words = page.extract_words()

            def in_any_table(word):
                cx = (word["x0"] + word["x1"]) / 2
                cy = (word["top"] + word["bottom"]) / 2
                for (x0, top, x1, bottom) in table_bboxes:
                    if x0 <= cx <= x1 and top <= cy <= bottom:
                        return True
                return False

            kept_words = [w for w in words if not in_any_table(w)]

            # Reconstruct text line by line (group words with close 'top' values)
            kept_words.sort(key=lambda w: (round(w["top"]), w["x0"]))
            lines = []
            current_line = []
            current_top = None
            for w in kept_words:
                rounded_top = round(w["top"])
                if current_top is None or abs(rounded_top - current_top) <= 2:
                    current_line.append(w["text"])
                    current_top = rounded_top if current_top is None else current_top
                else:
                    lines.append(" ".join(current_line))
                    current_line = [w["text"]]
                    current_top = rounded_top
            if current_line:
                lines.append(" ".join(current_line))

            pages_out.append(
                {
                    "page_number": page_number,
                    "text": "\n".join(lines),
                    "source_file": source_file,
                }
            )

    return pages_out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python table_extractor.py <path_to_pdf>")
        sys.exit(1)

    images = extract_table_regions_as_images(sys.argv[1], output_dir="../data/extracted_images")
    print(f"Rasterized {len(images)} table regions as images:")
    for im in images:
        print(f"  {im.image_id} -> {im.file_path} ({im.width}x{im.height})")
