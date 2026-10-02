from src.parse import PAGE_NUMBER_RE, clean_page, find_repeated_lines


def test_find_repeated_lines_flags_common_header_footer() -> None:
    pages = [
        "MCC LAWS OF CRICKET\nLAW 1 THE PLAYERS\nbody text one",
        "MCC LAWS OF CRICKET\nLAW 2 THE UMPIRES\nbody text two",
        "MCC LAWS OF CRICKET\nLAW 3 THE SCORERS\nbody text three",
    ]
    repeated = find_repeated_lines(pages)
    assert "MCC LAWS OF CRICKET" in repeated
    assert "LAW 1 THE PLAYERS" not in repeated
    assert "body text one" not in repeated


def test_clean_page_removes_header_and_page_numbers() -> None:
    repeated = {"MCC LAWS OF CRICKET"}
    page_text = "12\nMCC LAWS OF CRICKET\nLAW 1 THE PLAYERS\nSome real clause text."
    cleaned = clean_page(page_text, repeated)
    assert "MCC LAWS OF CRICKET" not in cleaned
    assert "12" not in cleaned.split("\n")
    assert "LAW 1 THE PLAYERS" in cleaned
    assert "Some real clause text." in cleaned


def test_page_number_regex_matches_common_forms() -> None:
    for text in ["12", "Page 12", "page 12", "12 of 90", "12/90"]:
        assert PAGE_NUMBER_RE.match(text), text


def test_page_number_regex_does_not_match_clause_numbers() -> None:
    for text in ["12.1", "LAW 12", "12.1 The bat"]:
        assert not PAGE_NUMBER_RE.match(text), text
