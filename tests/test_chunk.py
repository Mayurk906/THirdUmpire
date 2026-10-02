from src.chunk import (
    APPENDIX_HEADING_RE,
    APPENDIX_LETTER_ONLY_RE,
    BARE_SECTION_INLINE_RE,
    CLAUSE_NUMERIC_RE,
    DOT_LEADER_RE,
    LAW_HEADING_RE,
    SUBCLAUSE_NUMERIC_RE,
    build_chunks,
    is_titleish_upper,
    normalize_line,
    parse_clauses,
)

SOURCE = {
    "title": "Test Rules Document",
    "body": "mcc",
    "competition": "laws",
    "doc_type": "laws",
    "effective_from": "2026-01-01",
}

SYNTHETIC_PAGES = [
    {"doc_id": "test_doc", "page": 1, "text": "Some preamble text before any law starts."},
    {
        "doc_id": "test_doc",
        "page": 2,
        "text": "LAW 1 THE PLAYERS\n1.1\xa0\xa0Number of players\nA match is played between two sides.",
    },
    {
        "doc_id": "test_doc",
        "page": 3,
        "text": (
            "LAW 2 THE UMPIRES\n2.1\nAppointment of umpires\nTwo umpires shall control a match.\n"
            "2.1.1\nEach umpire stands at one end."
        ),
    },
    {
        # A table-of-contents-style repeat: same "LAW n Title" shape, but
        # Title Case instead of ALL-CAPS -- must not be treated as a heading.
        "doc_id": "test_doc",
        "page": 4,
        "text": "LAW 1 The players ....................................................... 4\nLAW 2 The umpires ....................................................... 7",
    },
    {
        "doc_id": "test_doc",
        "page": 5,
        "text": (
            "APPENDIX A\nDEFINITIONS AND EXPLANATIONS\nA.1\xa0\xa0The match\n"
            "The game is used in these Laws as a general term."
        ),
    },
]


# --- Heading regex tests on real sample strings observed in the source PDFs ---


def test_law_heading_matches_real_uppercase_heading() -> None:
    m = LAW_HEADING_RE.match("LAW 32 CAUGHT")
    assert m and m.group(1) == "32" and is_titleish_upper(m.group(2))


def test_law_heading_title_case_toc_entry_is_not_uppercase() -> None:
    m = LAW_HEADING_RE.match("LAW 32 Caught")
    assert m is not None  # shape matches...
    assert not is_titleish_upper(m.group(2))  # ...but fails the ToC guard


def test_bare_section_inline_matches_icc_ipl_style() -> None:
    m = BARE_SECTION_INLINE_RE.match("38 RUN OUT")
    assert m and m.group(1) == "38" and m.group(2) == "RUN OUT"


def test_appendix_letter_only_matches_bare_heading() -> None:
    assert APPENDIX_LETTER_ONLY_RE.match("APPENDIX A")
    assert APPENDIX_LETTER_ONLY_RE.match("Appendix A")  # IPL's Title Case


def test_appendix_letter_only_does_not_match_inline_reference() -> None:
    # "See Appendix D." is a cross-reference, not a heading.
    assert not APPENDIX_LETTER_ONLY_RE.match("Appendix D.")


def test_appendix_heading_matches_icc_colon_style() -> None:
    m = APPENDIX_HEADING_RE.match("APPENDIX A: DEFINITIONS")
    assert m and m.group(1).upper() == "A" and m.group(2) == "DEFINITIONS"


def test_clause_numeric_matches_inline_title() -> None:
    m = CLAUSE_NUMERIC_RE.match("32.1 Out Caught")
    assert m and m.group(1) == "32.1" and m.group(2) == "Out Caught"


def test_subclause_numeric_requires_two_or_more_decimals() -> None:
    assert SUBCLAUSE_NUMERIC_RE.match("32.1.1")
    assert not SUBCLAUSE_NUMERIC_RE.match("32.1")


def test_dot_leader_matches_toc_line() -> None:
    assert DOT_LEADER_RE.search("32 BOWLED ....................... 63")


def test_normalize_line_strips_control_chars_and_collapses_whitespace() -> None:
    assert normalize_line("LAW 40\t \x07TIMED OUT") == "LAW 40 TIMED OUT"
    assert normalize_line("3.1\xa0\xa0Correctness of scores") == "3.1 Correctness of scores"


# --- End-to-end parse_clauses behaviour on a small synthetic document ---


def test_parse_clauses_assigns_law_no_and_splits_clauses() -> None:
    records = parse_clauses(SYNTHETIC_PAGES)
    by_clause = {r["clause"]: r for r in records}

    assert by_clause["1.1"]["law_no"] == 1
    assert by_clause["1.1"]["clause_title"] == "Number of players"
    assert by_clause["2.1"]["law_no"] == 2
    assert by_clause["2.1"]["clause_title"] == "Appointment of umpires"
    # The sub-clause (2.1.1) folds into its parent, not a separate chunk.
    assert "2.1.1" not in by_clause
    assert "Each umpire stands at one end." in by_clause["2.1"]["text"]


def test_parse_clauses_rejects_title_case_toc_duplicate() -> None:
    records = parse_clauses(SYNTHETIC_PAGES)
    # The ToC-style "LAW 1 The players" / "LAW 2 The umpires" page must not
    # reset law_no or spawn new sections; clause 1.1/2.1 keep their real text.
    by_clause = {r["clause"]: r for r in records}
    assert by_clause["1.1"]["law_no"] == 1
    assert by_clause["2.1"]["law_no"] == 2


def test_parse_clauses_captures_appendix() -> None:
    records = parse_clauses(SYNTHETIC_PAGES)
    by_clause = {r["clause"]: r for r in records}
    assert "A.1" in by_clause
    assert by_clause["A.1"]["law_no"] == -1
    assert "APPENDIX A" in by_clause["A.1"]["section_path"]


def test_parse_clauses_captures_preamble() -> None:
    records = parse_clauses(SYNTHETIC_PAGES)
    by_clause = {r["clause"]: r for r in records}
    assert "preamble" in by_clause
    assert "Some preamble text" in by_clause["preamble"]["text"]


# --- Chunk id / content hash determinism, and metadata type safety ---


def test_chunk_ids_and_hashes_are_deterministic() -> None:
    records = parse_clauses(SYNTHETIC_PAGES)
    chunks_a = build_chunks("test_doc", SOURCE, records)
    chunks_b = build_chunks("test_doc", SOURCE, records)

    ids_a = [c.chunk_id for c in chunks_a]
    ids_b = [c.chunk_id for c in chunks_b]
    hashes_a = [c.content_hash for c in chunks_a]
    hashes_b = [c.content_hash for c in chunks_b]

    assert ids_a == ids_b
    assert hashes_a == hashes_b
    assert len(ids_a) == len(set(ids_a))  # all unique


def test_chunk_metadata_has_no_none_values() -> None:
    records = parse_clauses(SYNTHETIC_PAGES)
    chunks = build_chunks("test_doc", SOURCE, records)
    assert chunks, "expected at least one chunk from the synthetic document"
    for chunk in chunks:
        for key, value in chunk.metadata().items():
            assert value is not None, key
            assert isinstance(value, (str, int, float, bool)), (key, type(value))
