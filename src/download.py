"""Fetch non-manual source PDFs from config/sources.yaml and verify them.

Usage: python -m src.download [--force]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests
import yaml

from src.config import BASE_DIR, configure_logging

logger = logging.getLogger(__name__)

SOURCES_PATH = BASE_DIR / "config" / "sources.yaml"
RAW_DIR = BASE_DIR / "data" / "raw"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
TIMEOUT_SECONDS = 60


def load_sources() -> list[dict]:
    with SOURCES_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dest_path(doc_id: str) -> Path:
    return RAW_DIR / f"{doc_id}.pdf"


def is_valid_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def download_one(source: dict, force: bool) -> bool:
    doc_id = source["doc_id"]
    dest = dest_path(doc_id)

    if source.get("manual"):
        if dest.exists() and is_valid_pdf(dest):
            logger.info("%s: manual download present (%s)", doc_id, dest)
            return True
        if dest.exists():
            logger.error("%s: file exists at %s but doesn't look like a PDF", doc_id, dest)
            return False
        logger.error(
            "%s: manual download required. Save the PDF to %s (see PROJECT_PLAN.md for where to find it).",
            doc_id,
            dest,
        )
        return False

    if dest.exists() and not force:
        if is_valid_pdf(dest):
            logger.info("%s: already downloaded, skipping (%s)", doc_id, dest)
            return True
        logger.warning("%s: existing file isn't a valid PDF, re-downloading", doc_id)

    url = source["url"]
    logger.info("%s: downloading from %s", doc_id, url)
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.error("%s: download failed: %s", doc_id, exc)
        return False

    if response.status_code != 200:
        logger.error("%s: download failed with HTTP %s. Not guessing an alternative URL.", doc_id, response.status_code)
        return False

    if not response.content.startswith(b"%PDF-"):
        logger.error("%s: downloaded content doesn't look like a PDF", doc_id)
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    logger.info("%s: saved to %s (%d bytes)", doc_id, dest, len(response.content))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if the file already exists")
    args = parser.parse_args()

    configure_logging()
    sources = load_sources()

    all_ok = True
    for source in sources:
        if not download_one(source, args.force):
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
