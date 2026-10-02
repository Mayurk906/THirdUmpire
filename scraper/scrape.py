"""Discover new documents from listing pages and feed them into the existing
download.py -> ingest.py path (PROJECT_PLAN.md Phase 4).

Usage:
  python -m scraper.scrape                 # dry run: discover and print only
  python -m scraper.scrape --apply         # write sources.yaml, download, ingest

Never touches eval/golden.jsonl or eval/golden_multiturn.jsonl. Only adds or
updates entries in config/sources.yaml (marking outdated ones "superseded",
never deleting them) and calls the same src.download/src.ingest modules a
human would run by hand.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone

import requests
import yaml

from scraper.extract import extract_links
from scraper.manifest import has_changed, load_manifest, record, save_manifest, sha256_bytes
from scraper.metadata import make_doc_id, parse_effective_date, parse_effective_date_from_url, parse_format, parse_gender
from scraper.politeness import MIN_DELAY_SECONDS, USER_AGENT, is_allowed, polite_delay
from scraper.supersede import apply_supersede
from src.config import BASE_DIR, configure_logging

logger = logging.getLogger(__name__)

LISTING_PAGES_PATH = BASE_DIR / "config" / "listing_pages.yaml"
SOURCES_PATH = BASE_DIR / "config" / "sources.yaml"
TIMEOUT_SECONDS = 60

IPL_HUB_URL = "https://www.iplt20.com/documents"
# (title substring to match on the hub page, doc_type to assign)
IPL_SUBPAGE_TARGETS = [
    ("match playing conditions", "playing_conditions"),
    ("code of conduct for players", "code_of_conduct"),
]
IPL_EXCLUDE_KEYWORDS = ["ticket", "schedule", "terms and conditions", "anti corruption", "anti-corruption"]


def load_listing_pages() -> list[dict]:
    with LISTING_PAGES_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def load_sources() -> list[dict]:
    if not SOURCES_PATH.exists():
        return []
    with SOURCES_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def save_sources(sources: list[dict]) -> None:
    with SOURCES_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(sources, f, sort_keys=False, allow_unicode=True)


def fetch(url: str) -> str | None:
    if not is_allowed(url):
        logger.warning("%s: disallowed by robots.txt, skipping", url)
        return None
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.error("%s: fetch failed: %s", url, exc)
        return None
    if response.status_code != 200:
        logger.error("%s: fetch failed with HTTP %s. Not guessing an alternative URL.", url, response.status_code)
        return None
    response.encoding = response.encoding or "utf-8"
    return response.text


def _passes_keywords(title: str, include_keywords: list[str], exclude_keywords: list[str]) -> bool:
    lower = title.lower()
    if include_keywords and not any(k.lower() in lower for k in include_keywords):
        return False
    if any(k.lower() in lower for k in exclude_keywords):
        return False
    return True


def classify(title: str, url: str, body: str, competition_prefix: str, doc_type: str) -> dict | None:
    """Returns a config/sources.yaml-shaped dict, or None if the title
    doesn't name a recognisable format+date (e.g. an addendum/FAQ page)."""
    fmt = parse_format(title)
    if fmt is None:
        return None
    effective_from = parse_effective_date(title) or parse_effective_date_from_url(url)
    if effective_from is None:
        return None
    gender = parse_gender(title)
    competition = f"{competition_prefix}_{fmt}"
    doc_id = make_doc_id(body, gender, competition, effective_from)
    return {
        "doc_id": doc_id,
        "title": title.strip(),
        "body": body,
        "competition": competition,
        "gender": gender,
        "doc_type": doc_type,
        "effective_from": effective_from,
        "url": url,
        "manual": False,
    }


def discover_from_listing_page(page: dict) -> list[dict]:
    logger.info("Fetching listing page %r: %s", page["name"], page["url"])
    html = fetch(page["url"])
    if html is None:
        return []
    candidates = extract_links(html, page["parser"])
    logger.info("%s: %d raw links found", page["name"], len(candidates))

    discovered: list[dict] = []
    for title, url in candidates:
        if not _passes_keywords(title, page.get("include_keywords") or [], page.get("exclude_keywords") or []):
            continue
        source = classify(title, url, page["body"], page["competition_prefix"], page["doc_type"])
        if source is None:
            logger.debug("%s: skipped (no format/date match): %r", page["name"], title)
            continue
        discovered.append(source)
    return discovered


def discover_ipl_documents() -> list[dict]:
    """IPL's documents live one hop behind a hub page: each row on
    iplt20.com/documents links to its own sub-page (e.g. /playing-conditions),
    which embeds the actual PDF the same way BCCI's page does -- see
    config/listing_pages.yaml's module docstring for why this isn't a plain
    single-page listing_pages.yaml entry."""
    logger.info("Fetching IPL documents hub: %s", IPL_HUB_URL)
    html = fetch(IPL_HUB_URL)
    if html is None:
        return []

    from urllib.parse import urljoin

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    subpage_urls: dict[str, str] = {}  # doc_type -> absolute subpage url
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        for keyword, doc_type in IPL_SUBPAGE_TARGETS:
            if keyword in text and doc_type not in subpage_urls:
                # The hub page's links are relative (e.g. "/playing-conditions").
                subpage_urls[doc_type] = urljoin(IPL_HUB_URL, a["href"])

    discovered: list[dict] = []
    for doc_type, subpage_url in subpage_urls.items():
        polite_delay(MIN_DELAY_SECONDS)
        logger.info("Fetching IPL sub-page (%s): %s", doc_type, subpage_url)
        html = fetch(subpage_url)
        if html is None:
            continue
        for title, url in extract_links(html, "bcci_embedded_json"):
            lower = title.lower()
            if any(k in lower for k in IPL_EXCLUDE_KEYWORDS):
                continue
            effective_from = parse_effective_date(title) or parse_effective_date_from_url(url)
            if effective_from is None:
                logger.debug("ipl_documents: skipped (no date found): %r", title)
                continue
            # IPL's existing doc_ids (ipl_2026_pc, and the planned
            # ipl_2026_coc) follow a {competition}_{year}_{short doc_type}
            # shape, not make_doc_id()'s gender+format shape (which is
            # designed for ICC/BCCI domestic and would collide here: two IPL
            # docs sharing a year but different doc_types would otherwise
            # get the same doc_id).
            doc_type_suffix = "pc" if doc_type == "playing_conditions" else "coc"
            doc_id = f"ipl_{effective_from}_{doc_type_suffix}"
            discovered.append(
                {
                    "doc_id": doc_id,
                    "title": f"TATA IPL {effective_from} {title}".strip(),
                    "body": "bcci",
                    "competition": "ipl",
                    "gender": "all",
                    "doc_type": doc_type,
                    "effective_from": effective_from,
                    "url": url,
                    "manual": False,
                }
            )
    return discovered


def merge_sources(existing: list[dict], discovered: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (merged_sources, newly_added, refreshed).

    A `manual: true` existing entry is never touched (e.g.
    icc_mens_t20i_2025_07 stays the hand-added manual entry even though the
    scraper can now find it too -- manual means a human deliberately chose
    it). A non-manual existing entry whose URL has changed IS updated in
    place and reported as "refreshed" -- these sites rotate document URLs
    (confirmed: the ipl_2026_pc URL already in sources.yaml is dead), and
    since these entries were auto-discovered rather than hand-verified,
    there's no manual judgement to preserve by leaving a dead link in place.
    """
    existing_by_id = {s["doc_id"]: s for s in existing}
    new_entries: list[dict] = []
    refreshed: list[dict] = []
    seen: set[str] = set()

    for entry in discovered:
        doc_id = entry["doc_id"]
        if doc_id in seen:
            # A discovered run can itself repeat a doc_id (e.g. a
            # mislabelled duplicate link on the source page); keep the
            # first occurrence.
            logger.warning("%s: duplicate discovery, keeping first occurrence", doc_id)
            continue
        seen.add(doc_id)

        current = existing_by_id.get(doc_id)
        if current is None:
            new_entries.append(entry)
            existing_by_id[doc_id] = entry
        elif not current.get("manual") and current.get("url") != entry["url"]:
            updated = {**current, "url": entry["url"], "title": entry["title"]}
            existing_by_id[doc_id] = updated
            refreshed.append(updated)
        # else: manual entry, or URL unchanged -- leave as-is.

    # Preserve original order for pre-existing doc_ids; append new ones.
    merged = [existing_by_id[s["doc_id"]] for s in existing] + new_entries
    return merged, new_entries, refreshed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write sources.yaml and run download+ingest (default: dry run)")
    parser.add_argument("--skip-ipl", action="store_true", help="skip the IPL hub crawl")
    args = parser.parse_args()

    configure_logging()

    pages = load_listing_pages()
    discovered: list[dict] = []
    for page in pages:
        discovered.extend(discover_from_listing_page(page))
        polite_delay(MIN_DELAY_SECONDS)

    if not args.skip_ipl:
        discovered.extend(discover_ipl_documents())

    existing = load_sources()
    merged, new_entries, refreshed = merge_sources(existing, discovered)
    superseded = apply_supersede(merged)
    # apply_supersede() returns fresh copies; re-resolve new_entries/refreshed
    # (which still point at merge_sources()'s objects) against those copies
    # so downstream code sees the final status field.
    superseded_by_id = {s["doc_id"]: s for s in superseded}
    new_entries = [superseded_by_id[e["doc_id"]] for e in new_entries]
    refreshed = [superseded_by_id[e["doc_id"]] for e in refreshed]

    existing_status = {s["doc_id"]: s.get("status", "active") for s in existing}
    status_only = [
        s
        for s in superseded
        if s["doc_id"] in existing_status
        and s["status"] != existing_status[s["doc_id"]]
        and s["doc_id"] not in {e["doc_id"] for e in refreshed}
    ]

    n_sources = len(pages) + (0 if args.skip_ipl else 1)
    print(f"\nDiscovered {len(discovered)} candidate document(s) across {n_sources} source(s).")
    print(f"New to config/sources.yaml: {len(new_entries)}")
    for e in new_entries:
        print(f"  + {e['doc_id']}  [{e['competition']}, {e['gender']}, {e['effective_from']}]  {e['title']}")
    if refreshed:
        print(f"URL refreshed on existing (non-manual) documents: {len(refreshed)}")
        for e in refreshed:
            print(f"  ~ {e['doc_id']}  (new url)")
    if status_only:
        print(f"Status changes on existing documents: {len(status_only)}")
        for s in status_only:
            print(f"  ~ {s['doc_id']} -> {s['status']}")

    if not args.apply:
        print("\nDry run (no changes made). Re-run with --apply to write config/sources.yaml and ingest.")
        return 0

    if not new_entries and not refreshed and not status_only:
        print("\nNothing new; config/sources.yaml and the index are already up to date.")
        return 0

    save_sources(superseded)
    logger.info("Wrote %d source(s) to %s", len(superseded), SOURCES_PATH)

    manifest = load_manifest()
    to_download = new_entries + refreshed
    for source in to_download:
        polite_delay(MIN_DELAY_SECONDS)
        _download_and_record(source, manifest, force=source in refreshed)
    save_manifest(manifest)

    affected_doc_ids = sorted({s["doc_id"] for s in to_download} | {s["doc_id"] for s in status_only})
    if affected_doc_ids:
        logger.info("Running ingest for: %s", affected_doc_ids)
        result = subprocess.run([sys.executable, "-m", "src.ingest", *affected_doc_ids], cwd=str(BASE_DIR))
        if result.returncode != 0:
            logger.error("src.ingest failed with exit code %d", result.returncode)
            return 1

    return 0


def _download_and_record(source: dict, manifest: dict[str, dict], force: bool = False) -> None:
    from src.download import dest_path, download_one, is_valid_pdf

    ok = download_one(source, force=force)
    if not ok:
        return
    path = dest_path(source["doc_id"])
    if not is_valid_pdf(path):
        return
    sha256 = sha256_bytes(path.read_bytes())
    if has_changed(manifest, source["url"], sha256):
        record(
            manifest,
            source["url"],
            source["doc_id"],
            sha256,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            status=source.get("status", "active"),
        )


if __name__ == "__main__":
    raise SystemExit(main())
