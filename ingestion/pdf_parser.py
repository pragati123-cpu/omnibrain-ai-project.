"""
pdf_parser.py
--------------
Step 1 of the OmniBrain ingestion pipeline.

Parses a corporate financial PDF and extracts:
  1. Text content, per page
  2. Embedded raster images, saved to disk with page-level metadata

Why PyMuPDF (fitz)?
  - Fast, pure-Python-friendly, no external binary dependency at runtime
  - Gives page-level text AND lets us pull embedded images with their
    xref, so we can trace every image back to its exact source page
    (needed later for citations in the investment memo).

Note on tables/charts: this script extracts *raster* images only.
Charts drawn as vector graphics (common in matplotlib/Excel exports)
won't be picked up by this method — that's a known limitation to flag
in your Week 1 notes, and can be handled later by rasterizing full
pages for the Vision Agent.
"""

import fitz  # PyMuPDF
import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PageText:
    page_number: int          # 1-indexed
    text: str
    source_file: str


@dataclass
class ExtractedImage:
    image_id: str              # unique id, e.g. "docname_p3_img0"
    page_number: int
    file_path: str              # where the extracted image is saved
    source_file: str
    width: int
    height: int


def extract_text(doc: fitz.Document, source_file: str) -> list[PageText]:
    """Extract plain text from every page of the PDF."""
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")  # reading-order text extraction
        pages.append(PageText(page_number=i + 1, text=text, source_file=source_file))
    return pages


def extract_images(
    doc: fitz.Document,
    source_file: str,
    output_dir: str,
    min_size_px: int = 100,
) -> list[ExtractedImage]:
    """
    Extract embedded raster images from every page.

    min_size_px filters out tiny decorative/background images
    (logos, bullet icons, masks) that aren't meaningful chart/figure
    content — a common gotcha with pdfimages-style extraction.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc_name = Path(source_file).stem
    extracted = []

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)

                # Convert CMYK / other non-RGB color spaces to RGB
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                if pix.width < min_size_px or pix.height < min_size_px:
                    continue  # skip tiny decorative images

                image_id = f"{doc_name}_p{page_number}_img{img_index}"
                file_path = os.path.join(output_dir, f"{image_id}.png")
                pix.save(file_path)

                extracted.append(
                    ExtractedImage(
                        image_id=image_id,
                        page_number=page_number,
                        file_path=file_path,
                        source_file=source_file,
                        width=pix.width,
                        height=pix.height,
                    )
                )
            except Exception as e:
                print(f"  [warn] failed to extract image xref={xref} on page {page_number}: {e}")

    return extracted


def parse_pdf(pdf_path: str, image_output_dir: str) -> dict:
    """
    Full parse of one PDF: returns text-per-page and extracted images.
    This is the single entry point the rest of the pipeline calls.
    """
    doc = fitz.open(pdf_path)
    source_file = os.path.basename(pdf_path)

    print(f"Parsing '{source_file}' ({len(doc)} pages)...")

    pages = extract_text(doc, source_file)
    images = extract_images(doc, source_file, image_output_dir)

    print(f"  -> extracted text from {len(pages)} pages")
    print(f"  -> extracted {len(images)} images (after filtering tiny/decorative ones)")

    doc.close()
    return {
        "source_file": source_file,
        "pages": [asdict(p) for p in pages],
        "images": [asdict(im) for im in images],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    result = parse_pdf(sys.argv[1], image_output_dir="../data/extracted_images")

    out_json = "../data/parsed_output.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved parsed output to {out_json}")
