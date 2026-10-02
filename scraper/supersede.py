"""Mark outdated documents as superseded (PROJECT_PLAN.md Phase 4).

Within each (body, competition, gender, doc_type) group, the document with
the latest `effective_from` is "active" and every other document in that
group is "superseded" -- retrieval excludes superseded documents by default
(src/retrieve.py always filters status == "active").

Pure and testable: operates on plain source dicts (the same shape as
config/sources.yaml entries), not on the YAML file or Chroma directly --
scraper/scrape.py wires those up and re-syncs the index afterwards.
"""
from __future__ import annotations


def group_key(source: dict) -> tuple[str, str, str, str]:
    return (source["body"], source["competition"], source.get("gender", "all"), source["doc_type"])


def apply_supersede(sources: list[dict]) -> list[dict]:
    """Returns a new list (same order as `sources`) with `status` set to
    "active" for the newest document in each group and "superseded" for
    every other document in that group. A group of size 1 is always active.

    `effective_from` strings ("YYYY", "YYYY-MM" or "YYYY-MM-DD") compare
    correctly as plain strings here since they're all zero-padded, left-
    anchored at the year, and a shorter (less specific) value is always
    earlier-or-equal to a more specific value sharing the same prefix.
    """
    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for i, source in enumerate(sources):
        groups.setdefault(group_key(source), []).append(i)

    result = [dict(s) for s in sources]
    for indices in groups.values():
        newest_idx = max(indices, key=lambda i: sources[i]["effective_from"])
        for i in indices:
            result[i]["status"] = "active" if i == newest_idx else "superseded"
    return result
