from scraper.scrape import _passes_keywords, classify, merge_sources


# --- _passes_keywords ---


def test_passes_keywords_no_filters_always_true() -> None:
    assert _passes_keywords("Anything", [], [])


def test_passes_keywords_include_requires_match() -> None:
    assert _passes_keywords("Playing Conditions for Men's T20 Matches", ["playing conditions for"], [])
    assert not _passes_keywords("Men's Standard T20I Playing Conditions", ["playing conditions for"], [])


def test_passes_keywords_exclude_rejects_match() -> None:
    assert not _passes_keywords("ICC Men's T20 World Cup 2026 Playing Conditions", [], ["world cup"])


def test_passes_keywords_case_insensitive() -> None:
    assert _passes_keywords("MEN'S WORLD CUP PLAYING CONDITIONS", [], ["world cup"]) is False


# --- classify ---


def test_classify_icc_test_match() -> None:
    source = classify(
        "ICC Men's Test Match Playing Conditions - June 2025",
        "https://images.icc-cricket.com/image/upload/prd/lm8owaz03i86m1eneb7m.pdf",
        "icc",
        "icc",
        "playing_conditions",
    )
    assert source is not None
    assert source["doc_id"] == "icc_mens_test_2025_06"
    assert source["competition"] == "icc_test"
    assert source["gender"] == "men"
    assert source["effective_from"] == "2025-06"
    assert source["manual"] is False


def test_classify_returns_none_when_no_format() -> None:
    assert classify("IPL 2026 Code of Conduct", "https://x/y.pdf", "bcci", "ipl", "code_of_conduct") is None


def test_classify_falls_back_to_url_date_when_title_has_none() -> None:
    source = classify(
        "Playing Conditions for Men's Multi-day Matches",
        "https://www.bcci.tv/cms/files/x/y/1772193467492_2025-26_BCCI_Men_Multi_Day_PC.pdf",
        "bcci",
        "bcci_domestic",
        "playing_conditions",
    )
    assert source is not None
    assert source["effective_from"] == "2025"
    assert source["competition"] == "bcci_domestic_multiday"
    assert source["doc_id"] == "bcci_mens_domestic_multiday_2025"


# --- merge_sources ---


def _existing():
    return [
        {"doc_id": "a", "url": "https://old/a.pdf", "manual": False, "title": "A"},
        {"doc_id": "b", "url": "https://old/b.pdf", "manual": True, "title": "B (manual)"},
    ]


def test_merge_sources_adds_new_entry() -> None:
    discovered = [{"doc_id": "c", "url": "https://new/c.pdf", "title": "C"}]
    merged, new_entries, refreshed = merge_sources(_existing(), discovered)
    assert [s["doc_id"] for s in new_entries] == ["c"]
    assert refreshed == []
    assert {s["doc_id"] for s in merged} == {"a", "b", "c"}


def test_merge_sources_refreshes_dead_url_on_non_manual_entry() -> None:
    discovered = [{"doc_id": "a", "url": "https://fresh/a.pdf", "title": "A (new title)"}]
    merged, new_entries, refreshed = merge_sources(_existing(), discovered)
    assert new_entries == []
    assert len(refreshed) == 1
    assert refreshed[0]["url"] == "https://fresh/a.pdf"
    updated_a = next(s for s in merged if s["doc_id"] == "a")
    assert updated_a["url"] == "https://fresh/a.pdf"


def test_merge_sources_never_touches_manual_entry() -> None:
    discovered = [{"doc_id": "b", "url": "https://fresh/b.pdf", "title": "B (new title)"}]
    merged, new_entries, refreshed = merge_sources(_existing(), discovered)
    assert new_entries == []
    assert refreshed == []
    unchanged_b = next(s for s in merged if s["doc_id"] == "b")
    assert unchanged_b["url"] == "https://old/b.pdf"


def test_merge_sources_leaves_unchanged_url_alone() -> None:
    discovered = [{"doc_id": "a", "url": "https://old/a.pdf", "title": "A"}]
    merged, new_entries, refreshed = merge_sources(_existing(), discovered)
    assert new_entries == []
    assert refreshed == []


def test_merge_sources_dedupes_repeated_doc_id_in_discovered() -> None:
    discovered = [
        {"doc_id": "c", "url": "https://new/c1.pdf", "title": "C1"},
        {"doc_id": "c", "url": "https://new/c2.pdf", "title": "C2"},
    ]
    merged, new_entries, refreshed = merge_sources(_existing(), discovered)
    assert len(new_entries) == 1
    assert new_entries[0]["url"] == "https://new/c1.pdf"


def test_merge_sources_preserves_existing_order() -> None:
    discovered = [{"doc_id": "c", "url": "https://new/c.pdf", "title": "C"}]
    merged, _, _ = merge_sources(_existing(), discovered)
    assert [s["doc_id"] for s in merged] == ["a", "b", "c"]
