"""data/manifest.json: tracks discovered documents by source URL, so
re-running the scraper downloads nothing when nothing has changed
(PROJECT_PLAN.md Phase 4 "Done when"). Keyed by the document's source URL.

Manifest entry shape: {url: {doc_id, sha256, etag, last_modified,
downloaded_at, status}}.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config import BASE_DIR

MANIFEST_PATH = BASE_DIR / "data" / "manifest.json"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict[str, dict], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def has_changed(manifest: dict[str, dict], url: str, sha256: str) -> bool:
    """True if `url` is new to the manifest or its content hash differs
    from what was last recorded -- i.e. it needs downloading/re-ingesting."""
    entry = manifest.get(url)
    return entry is None or entry.get("sha256") != sha256


def record(
    manifest: dict[str, dict],
    url: str,
    doc_id: str,
    sha256: str,
    downloaded_at: str,
    status: str = "active",
    etag: str | None = None,
    last_modified: str | None = None,
) -> None:
    manifest[url] = {
        "doc_id": doc_id,
        "sha256": sha256,
        "etag": etag,
        "last_modified": last_modified,
        "downloaded_at": downloaded_at,
        "status": status,
    }
