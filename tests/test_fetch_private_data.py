from __future__ import annotations

import io
import zipfile

from scripts import fetch_private_data as fpd


class _FakeResponse:
    def __init__(self, json_data=None, content=b"", status_code=200, text=""):
        self._json_data = json_data
        self.content = content
        self.status_code = status_code
        self.text = text
        self.ok = status_code < 400

    def json(self):
        return self._json_data


def test_noop_when_chroma_db_already_exists(monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    monkeypatch.setattr(fpd, "_CHROMA_DIR", chroma_dir)

    def _boom(*args, **kwargs):
        raise AssertionError("must not make any network call when chroma_db/ already exists")

    monkeypatch.setattr("requests.get", _boom, raising=False)
    fpd.fetch_private_data_if_needed()  # must return without touching requests


def test_noop_when_secrets_unset(monkeypatch, tmp_path):
    missing_dir = tmp_path / "chroma_db"  # doesn't exist
    monkeypatch.setattr(fpd, "_CHROMA_DIR", missing_dir)
    monkeypatch.setattr(fpd, "GH_DATA_REPO", None)
    monkeypatch.setattr(fpd, "GH_DATA_RELEASE_TAG", None)
    monkeypatch.setattr(fpd, "GH_DATA_TOKEN", None)

    def _boom(*args, **kwargs):
        raise AssertionError("must not make any network call when GH_DATA_* secrets are unset")

    monkeypatch.setattr("requests.get", _boom, raising=False)
    fpd.fetch_private_data_if_needed()  # must return quietly, not raise


def test_repo_url_instead_of_owner_slash_repo_raises_clear_error(monkeypatch, tmp_path):
    missing_dir = tmp_path / "chroma_db"
    monkeypatch.setattr(fpd, "_CHROMA_DIR", missing_dir)
    monkeypatch.setattr(fpd, "GH_DATA_REPO", "https://github.com/owner/repo")
    monkeypatch.setattr(fpd, "GH_DATA_RELEASE_TAG", "v1")
    monkeypatch.setattr(fpd, "GH_DATA_TOKEN", "fake-token")

    def _boom(*args, **kwargs):
        raise AssertionError("must not make a network call with a malformed GH_DATA_REPO")

    monkeypatch.setattr("requests.get", _boom, raising=False)

    try:
        fpd.fetch_private_data_if_needed()
        assert False, "expected a RuntimeError"
    except RuntimeError as exc:
        assert "owner/repo" in str(exc)


def test_github_error_response_surfaces_status_and_body(monkeypatch, tmp_path):
    missing_dir = tmp_path / "chroma_db"
    monkeypatch.setattr(fpd, "_CHROMA_DIR", missing_dir)
    monkeypatch.setattr(fpd, "GH_DATA_REPO", "owner/private-data-repo")
    monkeypatch.setattr(fpd, "GH_DATA_RELEASE_TAG", "v1")
    monkeypatch.setattr(fpd, "GH_DATA_TOKEN", "fake-token")

    monkeypatch.setattr(
        "requests.get",
        lambda *a, **kw: _FakeResponse(status_code=404, text='{"message": "Not Found"}'),
        raising=False,
    )

    try:
        fpd.fetch_private_data_if_needed()
        assert False, "expected a RuntimeError"
    except RuntimeError as exc:
        assert "404" in str(exc)
        assert "Not Found" in str(exc)


def test_downloads_and_extracts_when_missing_and_configured(monkeypatch, tmp_path):
    missing_dir = tmp_path / "chroma_db"
    monkeypatch.setattr(fpd, "_CHROMA_DIR", missing_dir)
    monkeypatch.setattr(fpd, "BASE_DIR", tmp_path)
    monkeypatch.setattr(fpd, "GH_DATA_REPO", "owner/private-data-repo")
    monkeypatch.setattr(fpd, "GH_DATA_RELEASE_TAG", "v1")
    monkeypatch.setattr(fpd, "GH_DATA_TOKEN", "fake-token")

    # Build a tiny real zip containing one file under chroma_db/.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("chroma_db/marker.txt", "hello")
    zip_bytes = buf.getvalue()

    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        if url.endswith("/releases/tags/v1"):
            return _FakeResponse(json_data={"assets": [{"url": "https://api.github.com/asset/123"}]})
        if url == "https://api.github.com/asset/123":
            assert headers["Accept"] == "application/octet-stream"
            return _FakeResponse(content=zip_bytes)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("requests.get", fake_get, raising=False)

    fpd.fetch_private_data_if_needed()

    assert (tmp_path / "chroma_db" / "marker.txt").read_text() == "hello"
    assert len(calls) == 2
