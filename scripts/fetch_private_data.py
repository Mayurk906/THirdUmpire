"""Fetches chroma_db/ and models/reranker-ft/ from a PRIVATE GitHub repo's
release asset, for a hosted deploy that has no local index (see README
"Deploying a private demo"). A no-op everywhere else.

The rule documents aren't ours to redistribute publicly (see README
"Documents" -- data/ and chroma_db/ are gitignored for exactly that reason),
so a public repo can never contain them. This script is how a *private*,
invite-only deployment gets them instead: a separate private GitHub repo
holds a zip of both directories as a release asset, downloaded here with a
read-only, repo-scoped token (GH_DATA_TOKEN) that is itself a Streamlit
Cloud secret, never committed anywhere.

Called once at app.py startup, before anything imports src.retrieve or
src.rerank (which would otherwise try to open a chroma_db/ that doesn't
exist yet). Safe to call on a machine that already has chroma_db/ -- it
does nothing in that case, which is every local dev run.
"""
from __future__ import annotations

import io
import logging
import zipfile

from src.config import BASE_DIR, GH_DATA_RELEASE_TAG, GH_DATA_REPO, GH_DATA_TOKEN

logger = logging.getLogger(__name__)

_CHROMA_DIR = BASE_DIR / "chroma_db"
_API_BASE = "https://api.github.com"


def _asset_download_url() -> str | None:
    """The release's single zip asset's private API download URL (not its
    public browser_download_url, which 404s on a private repo without the
    same auth header -- the release-asset API endpoint accepts the token
    directly and needs an explicit octet-stream Accept header)."""
    import requests

    url = f"{_API_BASE}/repos/{GH_DATA_REPO}/releases/tags/{GH_DATA_RELEASE_TAG}"
    headers = {"Authorization": f"Bearer {GH_DATA_TOKEN}", "Accept": "application/vnd.github+json"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    assets = response.json().get("assets", [])
    if not assets:
        raise RuntimeError(f"Release {GH_DATA_RELEASE_TAG!r} in {GH_DATA_REPO!r} has no assets.")
    return assets[0]["url"]  # the API asset URL, not assets[0]["browser_download_url"]


def fetch_private_data_if_needed() -> None:
    if _CHROMA_DIR.exists():
        return  # Local dev, or a container that already fetched it this run.
    if not (GH_DATA_REPO and GH_DATA_RELEASE_TAG and GH_DATA_TOKEN):
        logger.info("No GH_DATA_* secrets set and no local chroma_db/ -- nothing to fetch, nothing to query.")
        return

    import requests

    logger.info("No local chroma_db/ found; fetching the private data bundle from %s@%s ...", GH_DATA_REPO, GH_DATA_RELEASE_TAG)
    asset_url = _asset_download_url()
    headers = {"Authorization": f"Bearer {GH_DATA_TOKEN}", "Accept": "application/octet-stream"}
    response = requests.get(asset_url, headers=headers, timeout=300)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(BASE_DIR)
    logger.info("Private data bundle extracted to %s.", BASE_DIR)
