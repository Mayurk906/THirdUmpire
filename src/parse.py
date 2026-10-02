"""Extract per-page text from raw PDFs, stripping repeated headers/footers.

Usage: python -m src.parse [doc_id ...]   (default: every PDF in data/raw)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf

from src.config import BASE_DIR, configure_logging

logger = logging.getLogger(__name__)

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# A standalone page number, optionally as "Page 12" or "12 of 90" / "12/90".
PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d+\s*((of|/)\s*\d+)?\s*$", re.IGNORECASE)

HEADER_FOOTER_THRESHOLD = 0.5


def extract_pages(pdf_path: Path) -> list[str]:
    doc = pymupdf.open(pdf_path)
    return [page.get_text() for page in doc]


def find_repeated_lines(pages: list[str]) -> set[str]:
    """Lines appearing on more than HEADER_FOOTER_THRESHOLD of pages are headers/footers."""
    counts: Counter[str] = Counter()
    for page_text in pages:
        lines = {line.strip() for line in page_text.splitlines() if line.strip()}
        counts.update(lines)
    threshold = len(pages) * HEADER_FOOTER_THRESHOLD
    return {line for line, count in counts.items() if count > threshold}


def clean_page(page_text: str, repeated_lines: set[str]) -> str:
    kept = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in repeated_lines:
            continue
        if PAGE_NUMBER_RE.match(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def parse_doc(doc_id: str) -> list[dict]:
    pdf_path = RAW_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} not found; run 'python -m src.download' first")

    raw_pages = extract_pages(pdf_path)
    repeated_lines = find_repeated_lines(raw_pages)
    if repeated_lines:
        logger.info("%s: removing %d repeated header/footer line(s)", doc_id, len(repeated_lines))

    records = []
    for i, page_text in enumerate(raw_pages, start=1):
        cleaned = clean_page(page_text, repeated_lines)
        records.append({"doc_id": doc_id, "page": i, "text": cleaned})
    return records


def save_pages(doc_id: str, records: list[dict]) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{doc_id}.pages.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_ids", nargs="*", help="doc_ids to parse (default: every PDF in data/raw)")
    args = parser.parse_args()

    configure_logging()

    doc_ids = args.doc_ids or [p.stem for p in sorted(RAW_DIR.glob("*.pdf"))]
    if not doc_ids:
        logger.error("No PDFs found in %s", RAW_DIR)
        return 1

    for doc_id in doc_ids:
        records = parse_doc(doc_id)
        out_path = save_pages(doc_id, records)
        logger.info("%s: %d pages -> %s", doc_id, len(records), out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
