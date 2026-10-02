import hashlib

import pytest

from src.index import compute_index_version, existing_hashes, existing_metadata, get_collection, sync_index
from src.schemas import Chunk


def make_chunk(chunk_id: str, clause: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="test_doc",
        body="mcc",
        competition="laws",
        gender="all",
        doc_type="laws",
        effective_from="2026-01-01",
        law_no=1,
        clause=clause,
        clause_title="Title",
        section_path=f"LAW 1 > {clause}",
        page_start=1,
        page_end=1,
        chunk_index=0,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        status="active",
        text=text,
    )


def fake_embed_fn(texts: list[str]) -> list[list[float]]:
    # Deterministic 1-dim "embedding" so tests don't need a real model.
    return [[float(len(t))] for t in texts]


@pytest.fixture
def collection(tmp_path):
    return get_collection(chroma_dir=str(tmp_path / "chroma_db"))


def test_sync_index_embeds_new_chunks(collection) -> None:
    chunks = [make_chunk("test_doc::1.1::0", "1.1", "hello world")]
    stats = sync_index(chunks, collection, fake_embed_fn)
    assert stats == {"embedded": 1, "skipped": 0, "deleted": 0, "status_updated": 0}
    assert collection.count() == 1


def test_sync_index_skips_unchanged_chunks(collection) -> None:
    chunks = [make_chunk("test_doc::1.1::0", "1.1", "hello world")]
    sync_index(chunks, collection, fake_embed_fn)
    stats = sync_index(chunks, collection, fake_embed_fn)
    assert stats == {"embedded": 0, "skipped": 1, "deleted": 0, "status_updated": 0}


def test_sync_index_reembeds_modified_chunk(collection) -> None:
    chunks = [make_chunk("test_doc::1.1::0", "1.1", "hello world")]
    sync_index(chunks, collection, fake_embed_fn)

    modified = [make_chunk("test_doc::1.1::0", "1.1", "hello world, modified")]
    stats = sync_index(modified, collection, fake_embed_fn)
    assert stats == {"embedded": 1, "skipped": 0, "deleted": 0, "status_updated": 0}

    stored = existing_hashes(collection, ["test_doc"])
    assert stored["test_doc::1.1::0"] == modified[0].content_hash


def test_sync_index_deletes_removed_chunk(collection) -> None:
    chunks = [
        make_chunk("test_doc::1.1::0", "1.1", "hello world"),
        make_chunk("test_doc::1.2::0", "1.2", "second clause"),
    ]
    sync_index(chunks, collection, fake_embed_fn)

    remaining = [chunks[0]]
    stats = sync_index(remaining, collection, fake_embed_fn)
    assert stats == {"embedded": 0, "skipped": 1, "deleted": 1, "status_updated": 0}
    assert collection.count() == 1


def test_sync_index_updates_status_without_reembedding(collection) -> None:
    chunks = [make_chunk("test_doc::1.1::0", "1.1", "hello world")]
    sync_index(chunks, collection, fake_embed_fn)

    superseded = [make_chunk("test_doc::1.1::0", "1.1", "hello world")]
    superseded[0].status = "superseded"
    stats = sync_index(superseded, collection, fake_embed_fn)
    assert stats == {"embedded": 0, "skipped": 0, "deleted": 0, "status_updated": 1}

    stored = existing_metadata(collection, ["test_doc"])
    assert stored["test_doc::1.1::0"]["status"] == "superseded"
    # content_hash (and therefore the embedding) was untouched.
    assert stored["test_doc::1.1::0"]["content_hash"] == chunks[0].content_hash


def test_compute_index_version_changes_when_content_changes(collection) -> None:
    chunks = [make_chunk("test_doc::1.1::0", "1.1", "hello world")]
    sync_index(chunks, collection, fake_embed_fn)
    version_1 = compute_index_version(collection)

    modified = [make_chunk("test_doc::1.1::0", "1.1", "hello world, modified")]
    sync_index(modified, collection, fake_embed_fn)
    version_2 = compute_index_version(collection)

    assert version_1 != version_2


def test_compute_index_version_stable_when_unchanged(collection) -> None:
    chunks = [make_chunk("test_doc::1.1::0", "1.1", "hello world")]
    sync_index(chunks, collection, fake_embed_fn)
    version_1 = compute_index_version(collection)
    sync_index(chunks, collection, fake_embed_fn)
    version_2 = compute_index_version(collection)
    assert version_1 == version_2
