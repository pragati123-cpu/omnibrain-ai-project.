"""
pipeline.py
-----------
Your Week 1 deliverable: parse PDF -> extract embedded images ->
rasterize table regions as images -> chunk the remaining prose text.

This does NOT touch embeddings or Qdrant -- that's your teammate's
part. What this script produces is a clean, self-contained JSON
manifest that your teammate's embedding script can load directly.

    PDF --[pdf_parser]-------> embedded images (charts/figures)
        --[table_extractor]--> table regions, ALSO rasterized as images
        --[table_extractor]--> page text WITH table regions excluded
        --[chunker]----------> text chunks (prose only)
        --[this file]--------> manifest.json  (the handoff artifact)

Why tables are images, not structured/SQL data: per the OmniBrain
spec, "It parses tables using a Vision-Language Model (VLM)" and
Week 1's task says to embed "both modalities" (text + image) into
Qdrant -- there's no third structured-data modality for PDF tables.
Tables get treated exactly like charts: rasterized, embedded via
CLIP, and read by the VLM at query time. (The Text-to-SQL agent is
for external "historical stock data", not PDF tables.)

Table text is still excluded from the prose chunker -- not because
it becomes structured data, but because the table is now represented
as an image; leaving its text in a prose chunk too would just
duplicate/garble the same numbers.

Run:
    python pipeline.py /path/to/financial_report.pdf
"""

import sys
import os
import json
from dataclasses import asdict

from pdf_parser import extract_images
from chunker import chunk_pages
from table_extractor import extract_table_regions_as_images, extract_text_excluding_tables
import fitz

IMAGE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "extracted_images")
MANIFEST_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "manifest.json")


def run_pipeline(pdf_path: str) -> dict:
    source_file = os.path.basename(pdf_path)

    print("=" * 60)
    print("STEP 1: Extracting embedded images (charts/figures)")
    print("=" * 60)
    doc = fitz.open(pdf_path)
    embedded_images = extract_images(doc, source_file, IMAGE_OUTPUT_DIR)
    doc.close()
    print(f"  -> extracted {len(embedded_images)} embedded images")

    print("\n" + "=" * 60)
    print("STEP 2: Rasterizing table regions as images (for VLM reading)")
    print("=" * 60)
    table_images = extract_table_regions_as_images(pdf_path, IMAGE_OUTPUT_DIR)
    print(f"  -> rasterized {len(table_images)} table regions")

    print("\n" + "=" * 60)
    print("STEP 3: Extracting prose text (table regions excluded)")
    print("=" * 60)
    pages = extract_text_excluding_tables(pdf_path)
    print(f"  -> extracted text from {len(pages)} pages")

    print("\n" + "=" * 60)
    print("STEP 4: Chunking prose text")
    print("=" * 60)
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
    print(f"Produced {len(chunks)} text chunks from {len(pages)} pages")

    # Merge embedded images + table-region images into ONE image list --
    # they're the same modality (image), just different origins. The
    # "kind" field lets a downstream agent tell them apart if it ever needs to.
    #
    # file_path is normalized to be relative to the project root (the
    # folder containing "ingestion/" and "data/"), NOT an absolute path.
    # An absolute path only works on the machine that generated it --
    # your teammate needs a path that still resolves after they pull
    # this repo onto their own machine.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _relativize(image_dict):
        image_dict = dict(image_dict)
        image_dict["file_path"] = os.path.relpath(
            os.path.abspath(image_dict["file_path"]), project_root
        ).replace(os.sep, "/")
        return image_dict

    all_images = (
        [_relativize({**asdict(im), "kind": "embedded_figure"}) for im in embedded_images]
        + [_relativize({**asdict(im), "kind": "table_region"}) for im in table_images]
    )

    manifest = {
        "source_file": source_file,
        "num_pages": len(pages),
        "text_chunks": [asdict(c) for c in chunks],
        "images": all_images,
    }
    return manifest


def save_manifest(manifest: dict, output_path: str = MANIFEST_OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {output_path}")
    print(f"  -> {len(manifest['text_chunks'])} text chunks")
    print(f"  -> {len(manifest['images'])} images (embedded figures + rasterized tables)")
    print("\nHand this file + the extracted_images/ folder to the embedding step.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_pdf>")
        sys.exit(1)

    manifest = run_pipeline(sys.argv[1])
    save_manifest(manifest)
