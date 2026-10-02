from pathlib import Path

from scraper.manifest import has_changed, load_manifest, record, save_manifest, sha256_bytes


def test_load_manifest_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "manifest.json") == {}


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest: dict = {}
    record(manifest, "https://example.com/a.pdf", "doc_a", "abc123", "2026-01-01T00:00:00Z")
    save_manifest(manifest, path)

    loaded = load_manifest(path)
    assert loaded["https://example.com/a.pdf"]["doc_id"] == "doc_a"
    assert loaded["https://example.com/a.pdf"]["sha256"] == "abc123"


def test_has_changed_true_for_new_url() -> None:
    assert has_changed({}, "https://example.com/a.pdf", "abc123") is True


def test_has_changed_false_when_hash_matches() -> None:
    manifest: dict = {}
    record(manifest, "https://example.com/a.pdf", "doc_a", "abc123", "2026-01-01T00:00:00Z")
    assert has_changed(manifest, "https://example.com/a.pdf", "abc123") is False


def test_has_changed_true_when_hash_differs() -> None:
    manifest: dict = {}
    record(manifest, "https://example.com/a.pdf", "doc_a", "abc123", "2026-01-01T00:00:00Z")
    assert has_changed(manifest, "https://example.com/a.pdf", "def456") is True


def test_sha256_bytes_is_deterministic() -> None:
    assert sha256_bytes(b"hello") == sha256_bytes(b"hello")
    assert sha256_bytes(b"hello") != sha256_bytes(b"world")


def test_record_stores_optional_fields() -> None:
    manifest: dict = {}
    record(
        manifest,
        "https://example.com/a.pdf",
        "doc_a",
        "abc123",
        "2026-01-01T00:00:00Z",
        status="superseded",
        etag='"xyz"',
        last_modified="Mon, 01 Jan 2026 00:00:00 GMT",
    )
    entry = manifest["https://example.com/a.pdf"]
    assert entry["status"] == "superseded"
    assert entry["etag"] == '"xyz"'
    assert entry["last_modified"] == "Mon, 01 Jan 2026 00:00:00 GMT"
