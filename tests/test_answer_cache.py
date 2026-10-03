from __future__ import annotations

from src import answer_cache
from src.schemas import Answer, Chunk


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        body="laws",
        competition="laws",
        gender="all",
        doc_type="laws",
        effective_from="2026",
        law_no=1,
        clause="1.1",
        clause_title="Title",
        section_path="1 > 1.1",
        page_start=1,
        page_end=1,
        chunk_index=0,
        content_hash="hash",
        status="active",
        text="text",
    )


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(answer_cache, "CACHE_PATH", tmp_path / "answers.sqlite3")
    monkeypatch.setattr(answer_cache, "_index_version", lambda: "v1")


def test_miss_then_hit_roundtrip(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    chunks = [make_chunk("a")]
    key = answer_cache.cache_key("What is Law 1?", chunks)

    assert answer_cache.get(key) is None

    answer = Answer(answer="Eleven players.", found=True, applies_to="laws", citations=[])
    answer_cache.put(key, answer)

    cached = answer_cache.get(key)
    assert cached is not None
    assert cached.answer == "Eleven players."
    assert cached.found is True


def test_key_changes_with_question_text(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    chunks = [make_chunk("a")]
    assert answer_cache.cache_key("Question A", chunks) != answer_cache.cache_key("Question B", chunks)


def test_key_is_insensitive_to_case_and_whitespace(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    chunks = [make_chunk("a")]
    assert answer_cache.cache_key("What  is Law 1?", chunks) == answer_cache.cache_key("what is law 1?", chunks)


def test_key_changes_with_context_chunks(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert answer_cache.cache_key("Q", [make_chunk("a")]) != answer_cache.cache_key("Q", [make_chunk("b")])


def test_key_changes_with_index_version(monkeypatch, tmp_path):
    monkeypatch.setattr(answer_cache, "CACHE_PATH", tmp_path / "answers.sqlite3")
    chunks = [make_chunk("a")]

    monkeypatch.setattr(answer_cache, "_index_version", lambda: "v1")
    key_v1 = answer_cache.cache_key("Q", chunks)
    monkeypatch.setattr(answer_cache, "_index_version", lambda: "v2")
    key_v2 = answer_cache.cache_key("Q", chunks)

    assert key_v1 != key_v2
