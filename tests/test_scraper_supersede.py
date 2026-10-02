from scraper.supersede import apply_supersede, group_key


def make_source(doc_id, body="icc", competition="icc_test", gender="men", doc_type="playing_conditions", effective_from="2025"):
    return {
        "doc_id": doc_id,
        "body": body,
        "competition": competition,
        "gender": gender,
        "doc_type": doc_type,
        "effective_from": effective_from,
    }


def test_group_key_includes_all_four_fields() -> None:
    s = make_source("x", body="bcci", competition="bcci_domestic_t20", gender="women", doc_type="playing_conditions")
    assert group_key(s) == ("bcci", "bcci_domestic_t20", "women", "playing_conditions")


def test_single_document_group_is_active() -> None:
    sources = [make_source("icc_mens_test_2025_06", effective_from="2025-06")]
    result = apply_supersede(sources)
    assert result[0]["status"] == "active"


def test_newest_in_group_is_active_others_superseded() -> None:
    sources = [
        make_source("a", effective_from="2023-06"),
        make_source("b", effective_from="2025-06"),
        make_source("c", effective_from="2024-06"),
    ]
    result = apply_supersede(sources)
    by_id = {r["doc_id"]: r["status"] for r in result}
    assert by_id["a"] == "superseded"
    assert by_id["b"] == "active"
    assert by_id["c"] == "superseded"


def test_different_groups_are_independent() -> None:
    sources = [
        make_source("test_old", competition="icc_test", effective_from="2023"),
        make_source("test_new", competition="icc_test", effective_from="2025"),
        make_source("odi_only", competition="icc_odi", effective_from="2020"),
    ]
    result = apply_supersede(sources)
    by_id = {r["doc_id"]: r["status"] for r in result}
    assert by_id["test_old"] == "superseded"
    assert by_id["test_new"] == "active"
    assert by_id["odi_only"] == "active"  # sole member of its own group


def test_gender_distinguishes_groups() -> None:
    sources = [
        make_source("mens", gender="men", effective_from="2025"),
        make_source("womens", gender="women", effective_from="2025"),
    ]
    result = apply_supersede(sources)
    by_id = {r["doc_id"]: r["status"] for r in result}
    assert by_id["mens"] == "active"
    assert by_id["womens"] == "active"


def test_mixed_date_granularity_compares_correctly() -> None:
    # "2024-12" (Dec 2024) is later than "2024" (year-only) which is later
    # than "2023-06".
    sources = [
        make_source("year_only", effective_from="2024"),
        make_source("month_specific", effective_from="2024-12"),
        make_source("earlier", effective_from="2023-06"),
    ]
    result = apply_supersede(sources)
    by_id = {r["doc_id"]: r["status"] for r in result}
    assert by_id["month_specific"] == "active"
    assert by_id["year_only"] == "superseded"
    assert by_id["earlier"] == "superseded"


def test_preserves_original_order() -> None:
    sources = [
        make_source("b", effective_from="2025"),
        make_source("a", effective_from="2023"),
    ]
    result = apply_supersede(sources)
    assert [r["doc_id"] for r in result] == ["b", "a"]


def test_does_not_mutate_input() -> None:
    sources = [make_source("a", effective_from="2025")]
    apply_supersede(sources)
    assert "status" not in sources[0]
