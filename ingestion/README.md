# OmniBrain — Week 1: Parsing + Chunking + Image Extraction (incl. Tables)

**Your scope:** PDF → extracted images (embedded figures + rasterized
table regions) → chunked prose text → `manifest.json` handoff file.
**Not your scope (teammate's part):** embedding text/images and
storing them in Qdrant.

**Note on tables:** an earlier version of this pipeline extracted
tables as structured data for a SQL agent. That was corrected after
checking the full project spec — tables are meant to be treated as
**images**, read by a Vision-Language Model at query time, same as
charts. The spec's Text-to-SQL agent is for external historical stock
data, not PDF tables. See "Design decisions" below for the exact
reasoning.

## Structure

```
omnibrain/
├── ingestion/
│   ├── pdf_parser.py       # embedded image extraction (PyMuPDF)
│   ├── table_extractor.py  # detects tables (pdfplumber), rasterizes them as images (PyMuPDF)
│   ├── chunker.py          # split prose text into overlapping chunks
│   └── pipeline.py         # runs all steps, writes manifest.json
├── data/
│   ├── raw_pdfs/            # input PDFs go here
│   ├── extracted_images/    # embedded figures AND rasterized tables land here (PNG)
│   └── manifest.json        # <- the file you hand to your teammate
└── requirements_parsing.txt
```

## Setup

```bash
pip install -r requirements_parsing.txt
```

No API keys, no model downloads, no external services — this part
runs fully offline.

## Run

```bash
cd ingestion
python pipeline.py ../data/raw_pdfs/your_report.pdf
```

Produces `data/manifest.json` plus the extracted image files in
`data/extracted_images/`.

## The handoff contract (share this with your teammate)

`manifest.json` looks like:

```json
{
  "source_file": "demo_with_table.pdf",
  "num_pages": 2,
  "text_chunks": [
    {
      "chunk_id": "demo_with_table.pdf_p1_c0",
      "text": "Total revenue for fiscal year 2025 was $4.2 billion...",
      "page_number": 1,
      "source_file": "demo_with_table.pdf",
      "chunk_index_on_page": 0
    }
  ],
  "images": [
    {
      "image_id": "demo_with_table_p2_img0",
      "page_number": 2,
      "file_path": "data/extracted_images/demo_with_table_p2_img0.png",
      "source_file": "demo_with_table.pdf",
      "width": 480,
      "height": 360,
      "kind": "embedded_figure"
    },
    {
      "image_id": "demo_with_table_p2_table0",
      "page_number": 2,
      "file_path": "data/extracted_images/demo_with_table_p2_table0.png",
      "source_file": "demo_with_table.pdf",
      "width": 1104,
      "height": 294,
      "bbox": [72.0, 142.0, 432.0, 232.0],
      "kind": "table_region"
    }
  ]
}
```

Note: `file_path` is relative to the **project root** (the folder containing both `ingestion/` and `data/`), not to `ingestion/` where the script runs from. Your teammate should join it with the project root, not the current working directory — e.g. `os.path.join(project_root, image["file_path"])`.

Your teammate's embedding script just needs to:
1. Load `manifest.json`
2. For each item in `text_chunks`: embed `text`, upsert to Qdrant with
   the rest of the fields as payload (for citation/traceability).
3. For each item in `images`: open the image at `file_path`, embed it
   with CLIP, upsert to Qdrant with the rest of the fields as payload.
   The `kind` field (`embedded_figure` vs `table_region`) is there in
   case the VLM or Vision Agent ever needs to treat the two
   differently — but for embedding purposes they're the same modality.

There is no separate SQL/structured-data output from this pipeline.
The Text-to-SQL agent in the roadmap is scoped to external historical
stock data, not anything from this PDF.

This keeps your two halves decoupled — they can build/test the
embedding step against this manifest without ever running your PDF
parsing code, and you can iterate on parsing without touching their
Qdrant code.

## Design decisions worth knowing

- **PyMuPDF for embedded image extraction** — pulls raster images
  (charts/figures) with page-level + xref traceability, filters out
  tiny decorative images (logos, masks) automatically.
- **pdfplumber to detect tables, PyMuPDF to rasterize them** —
  `find_tables()` locates a table's bounding box via ruled lines /
  column alignment, then that region gets rendered at 3x zoom as a
  PNG (readable resolution for a VLM). This matches the spec directly:
  the use case says tables are parsed "using a Vision-Language Model",
  and Week 1's own task says to embed "both modalities" (text + image)
  — there's no third structured-data modality described anywhere in
  the project doc for PDF tables.
- **Why not extract tables as structured rows/SQL data instead?**
  That was the original plan, but it doesn't match the spec: the
  Text-to-SQL agent is explicitly scoped to external "historical stock
  data" (e.g. a market data source), not tables inside the PDF. Mixing
  those up would mean building an agent path that never gets used by
  the intended architecture.
- **Table regions are still excluded from prose text before
  chunking** — not because they become structured data, but because
  the table is now represented as an image; leaving its text in a
  prose chunk too would duplicate/garble the same numbers.
- **`RecursiveCharacterTextSplitter` for chunking** — splits on
  paragraph → sentence → word boundaries in that priority order.
  500 char chunks / 50 char overlap is a reasonable starting point —
  tune once you see chunk quality on the real 500-page document.
- **Known limitation (flag this to the team):** table detection relies
  on ruled lines/column alignment (pdfplumber's `find_tables`).
  Borderless tables or tables with irregular spacing may be missed —
  worth spot-checking real pages once you have the actual document.

## Tested with

Two synthetic PDFs:
- `demo_financial_report.pdf` — text + a generated bar chart image.
- `demo_with_table.pdf` — text + a real ruled table (quarterly
  revenue breakdown) + a chart image. Confirmed the table rasterizes
  as a crisp, readable PNG (visually verified), lands in the same
  `images` array as the embedded chart with `kind: "table_region"`,
  and its numbers do NOT also appear duplicated in the prose chunk
  for that page.

## Week 1 checklist (your part)

- [x] Parse PDF → embedded images
- [x] Rasterize table regions as images (for VLM reading, per spec)
- [x] Text-chunking mechanism (table regions excluded from prose)
- [x] Clean manifest.json handoff format for the embedding teammate
- [ ] Run against the actual 500-page target document
- [ ] Spot-check table detection accuracy on real pages (borderless tables may need extra handling)
- [ ] Sync with teammate on the manifest schema before they start their half
- [ ] Tune chunk_size/overlap once you see real chunk quality
