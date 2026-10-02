from scraper.metadata import (
    make_doc_id,
    parse_effective_date,
    parse_effective_date_from_url,
    parse_format,
    parse_gender,
)


# --- Real titles observed on the ICC and BCCI playing-conditions pages ---


def test_parse_gender_detects_women() -> None:
    assert parse_gender("ICC Women's Test Match playing Conditions - July 2026") == "women"


def test_parse_gender_detects_men() -> None:
    assert parse_gender("ICC Men's Test Match Playing Conditions - June 2025") == "men"


def test_parse_gender_defaults_to_all_when_unnamed() -> None:
    assert parse_gender("IPL 2026 Code of Conduct for Players and Team Officials") == "all"


def test_parse_format_test_match() -> None:
    assert parse_format("ICC Men's Test Match Playing Conditions - June 2025") == "test"


def test_parse_format_odi() -> None:
    assert parse_format("ICC Men's Standard ODI Playing Conditions Effective July 2025") == "odi"
    assert parse_format("Playing Conditions for Men's One-day Matches") == "odi"


def test_parse_format_t20i_is_distinguished_from_domestic_t20() -> None:
    assert parse_format("ICC Men's T20I Playing Conditions - July 2025") == "t20i"
    assert parse_format("Men's Twenty20 International Playing Conditions - Effective December 2023") == "t20i"
    # No "international"/"I" -> the domestic format, not t20i.
    assert parse_format("Playing Conditions for Men's T20 Matches") == "t20"


def test_parse_format_multiday() -> None:
    assert parse_format("Playing Conditions for Men's Multi-day Matches") == "multiday"


def test_parse_format_returns_none_for_non_format_titles() -> None:
    assert parse_format("IPL 2026 Code of Conduct for Players and Team Officials") is None
    assert parse_format("Playing Conditions FAQ") is None


def test_parse_effective_date_month_year() -> None:
    assert parse_effective_date("ICC Men's Test Match Playing Conditions - June 2025") == "2025-06"


def test_parse_effective_date_with_day() -> None:
    assert parse_effective_date("Women's Test Match Playing Conditions - Effective 1st January 2024") == "2024-01-01"


def test_parse_effective_date_year_only_fallback() -> None:
    assert parse_effective_date("IPL 2026 Code of Conduct for Players and Team Officials") == "2026"


def test_parse_effective_date_none_when_absent() -> None:
    assert parse_effective_date("Playing Conditions FAQ") is None


def test_parse_effective_date_from_url_extracts_season_start_year() -> None:
    url = "https://www.bcci.tv/cms/files/gtykgr9m2ngt/xxxx/1772193467492_2025-26_BCCI_Men_Multi_Day_PC.pdf"
    assert parse_effective_date_from_url(url) == "2025"


def test_parse_effective_date_from_url_none_when_no_season_pattern() -> None:
    url = "https://www.bcci.tv/cms/files/gtykgr9m2ngt/xxxx/yyyy/Mens_Standard_Test_Match_Playing_Conditions.pdf"
    assert parse_effective_date_from_url(url) is None


def test_parse_effective_date_from_url_extracts_plain_year() -> None:
    # IPL's filenames have no season range, just a bare year -- and a
    # leading ~13-digit millisecond timestamp prefix that must not be
    # mistaken for one.
    url = "https://www.iplt20.com/cms/files/jot2qw8s50z1/x/y/1775736835406_TATA_IPL_2026_Match_Playing_Conditions.pdf"
    assert parse_effective_date_from_url(url) == "2026"


def test_parse_effective_date_from_url_does_not_match_inside_timestamp_prefix() -> None:
    # A contrived filename whose timestamp prefix happens to contain "20xx"
    # digits with no real separator around them must not false-positive.
    url = "https://example.com/files/120261234567_Some_Document.pdf"
    assert parse_effective_date_from_url(url) is None


def test_make_doc_id_matches_existing_naming_convention() -> None:
    # icc_mens_t20i_2025_07 is the doc_id already used for the Phase 1 manual
    # download in config/sources.yaml -- the scraper must produce the same
    # shape for newly discovered documents.
    assert make_doc_id("icc", "men", "t20i", "2025-07") == "icc_mens_t20i_2025_07"
    assert make_doc_id("icc", "women", "test", "2026-07-01") == "icc_womens_test_2026_07_01"
    assert make_doc_id("bcci", "men", "domestic_multiday", "2023-12") == "bcci_mens_domestic_multiday_2023_12"
