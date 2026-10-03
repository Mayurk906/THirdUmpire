"""A tiny local disk cache for generate_answer(), keyed on everything that
can change its output.

The free Gemini tier caps GEMINI_MODEL at 20 requests/day (see
config.py's DEFAULT_FIRST_STAGE comment and README "Known issues") --
re-asking the same question under the same settings during a demo or a
debugging session shouldn't spend another one of them. Uses stdlib sqlite3
only: no new dependency, and it's a cache, not state anyone needs to read
back structurally, so a single key -> JSON blob table is enough.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from src.config import BASE_DIR, PROMPT_VERSION
from src.schemas import Answer, Chunk

CACHE_PATH = BASE_DIR / ".cache" / "answers.sqlite3"
_INDEX_VERSION_PATH = BASE_DIR / "chroma_db" / "index_version.txt"


def _index_version() -> str:
    try:
        return _INDEX_VERSION_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "unknown"


def cache_key(question: str, context_chunks: list[Chunk]) -> str:
    """Keyed on the question plus the exact context chunk_ids (which already
    encode competition/first_stage/reranker/compared-competitions -- however
    retrieval got to this context, the same context + question + prompt +
    index version will get the same answer), not retrieval settings
    directly. That also means a retrieval-side change (new chunk ranked in)
    correctly misses the cache even if every setting string is identical."""
    normalized_question = " ".join(question.strip().lower().split())
    chunk_ids = ",".join(c.chunk_id for c in context_chunks)
    raw = f"{normalized_question}|{chunk_ids}|{PROMPT_VERSION}|{_index_version()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, answer_json TEXT NOT NULL)")
    return conn


def get(key: str) -> Answer | None:
    try:
        with _connect() as conn:
            row = conn.execute("SELECT answer_json FROM answers WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error:
        return None  # A corrupt/locked cache file degrades to "no cache", not a crash.
    if row is None:
        return None
    try:
        return Answer.model_validate(json.loads(row[0]))
    except (json.JSONDecodeError, ValueError):
        return None


def put(key: str, answer: Answer) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO answers (key, answer_json) VALUES (?, ?)",
                (key, answer.model_dump_json()),
            )
    except sqlite3.Error:
        pass  # Caching is an optimization; a write failure must never break answering.
