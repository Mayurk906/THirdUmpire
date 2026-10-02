"""Clause-aware chunking: data/processed/{doc_id}.pages.jsonl -> data/processed/chunks.jsonl.

One chunk equals one clause (a single-decimal number, e.g. "32.1" or an
appendix's "A.1"). Deeper sub-clause numbers (e.g. "32.1.1") have no title of
their own and are folded into their parent clause's text. Text before the
first heading is kept as a single "preamble" chunk per document.

See PROJECT_PLAN.md Phase 1 step 4, and the Phase 1 stop-and-inspect notes
this was built from, for the reasoning behind each regex and filter below.

Usage: python -m src.chunk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import BASE_DIR, CHUNK_OVERLAP, MAX_CHUNK_CHARS, configure_logging
from src.schemas import Chunk

logger = logging.getLogger(__name__)

SOURCES_PATH = BASE_DIR / "config" / "sources.yaml"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# --- Noise filters (table of contents entries) ---

# A dot-leader ToC line, e.g. "32 BOWLED ....................... 63".
DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")

# --- Heading patterns ---

# MCC style: "LAW 32 CAUGHT" on one line. Table-of-contents entries use the
# same "LAW n Title" shape but in Title Case, and a back-of-book index page
# can repeat already-seen Law numbers in condensed form -- the caller checks
# the title is upper case AND that the number is higher than the last one
# seen before accepting this as a real heading.
LAW_HEADING_RE = re.compile(r"^LAW\s+(\d{1,2})\s+(.+)$")

# Appendix heading. Casing and separator vary by document: MCC uses a bare
# letter with an ALL-CAPS title wrapped onto the next 1-2 lines ("APPENDIX A"
# / "DEFINITIONS AND EXPLANATIONS" / "OF WORDS OR PHRASES NOT DEFINED IN THE
# TEXT"); ICC uses an inline colon ("APPENDIX A: DEFINITIONS"); IPL uses a
# bare letter with a single Title-Case line after it ("Appendix A" /
# "Definitions"). The caller requires a minimum title length (rejects the
# empty table-of-contents entries) and a strictly-increasing letter (rejects
# already-seen repeats from a condensed back-of-book index).
APPENDIX_HEADING_RE = re.compile(r"^APPENDIX\s+([A-Z])[:.]?\s+(.+)$", re.IGNORECASE)
# No trailing punctuation here: real bare headings are exactly "APPENDIX A"
# with nothing else, whereas wrapped inline cross-references like "See
# Appendix D." end in a period and must not be mistaken for a heading.
APPENDIX_LETTER_ONLY_RE = re.compile(r"^APPENDIX\s+([A-Z])$", re.IGNORECASE)
APPENDIX_TITLE_LOOKAHEAD = 2
MIN_APPENDIX_TITLE_LEN = 8

# ICC/IPL style top-level section, no "LAW" word, e.g. "38 RUN OUT" or a bare
# "38" with the title on the next line. Validated further by the caller with
# a strictly-increasing-numbering check, since a bare 1-2 digit number alone
# is a weak signal (it could be a year or a list marker in body prose). Note
# the very first section of a document may have no number at all (e.g. IPL's
# "THE PLAYERS"), so the first accepted candidate can be any number, not
# necessarily 1.
BARE_SECTION_INLINE_RE = re.compile(r"^(\d{1,2})\s+([A-Z][A-Z0-9 ,;'\-.&()]{2,80})$")
BARE_SECTION_NUMBER_ONLY_RE = re.compile(r"^(\d{1,2})$")

# Clause level: exactly one decimal point, e.g. "32.1" or "A.1".
CLAUSE_NUMERIC_RE = re.compile(r"^(\d{1,3}\.\d{1,3})(?:\s+(.+))?$")
CLAUSE_APPENDIX_RE = re.compile(r"^([A-Z]\.\d{1,3})(?:\s+(.+))?$")

# Sub-clause: two or more decimal points, e.g. "32.1.1" or "B.3.1.3". These
# have no title of their own and fold into the current clause's text.
SUBCLAUSE_NUMERIC_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){2,}\b")
SUBCLAUSE_APPENDIX_RE = re.compile(r"^[A-Z](?:\.\d{1,3}){2,}\b")

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\xa0]")
WHITESPACE_RE = re.compile(r"[ \t]+")


def normalize_line(line: str) -> str:
    """Strip control-character artifacts and collapse tabs/nbsp to single spaces."""
    cleaned = CONTROL_CHARS_RE.sub(" ", line)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def is_titleish_upper(text: str) -> bool:
    """True if text has at least one letter and no lower-case letters.

    Distinguishes real ALL-CAPS headings from table-of-contents entries,
    which render the same heading in Title Case.
    """
    return any(c.isalpha() for c in text) and text == text.upper()


def load_sources() -> dict[str, dict]:
    with SOURCES_PATH.open("r", encoding="utf-8") as f:
        sources = yaml.safe_load(f)
    return {s["doc_id"]: s for s in sources}


def load_pages(doc_id: str) -> list[dict]:
    path = PROCESSED_DIR / f"{doc_id}.pages.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@dataclass
class _ClauseBuffer:
    law_no: int
    clause: str
    clause_title: str
    section_path: str
    page_start: int
    lines: list[str] = field(default_factory=list)


def _flush(buf: _ClauseBuffer | None, page_end: int, out: list[dict]) -> None:
    if buf is None or not buf.lines:
        return
    out.append(
        {
            "law_no": buf.law_no,
            "clause": buf.clause,
            "clause_title": buf.clause_title,
            "section_path": buf.section_path,
            "page_start": buf.page_start,
            "page_end": page_end,
            "text": "\n".join(buf.lines),
        }
    )


def parse_clauses(pages: list[dict]) -> list[dict]:
    """Walk a document's pages and return one record per clause (or preamble)."""
    # Flatten to (page_number, normalized_line) pairs, dropping ToC/header noise.
    stream: list[tuple[int, str]] = []
    for page in pages:
        for raw_line in page["text"].split("\n"):
            if DOT_LEADER_RE.search(raw_line):
                continue
            line = normalize_line(raw_line)
            if not line:
                continue
            stream.append((page["page"], line))

    records: list[dict] = []
    preamble_lines: list[str] = []
    preamble_page_start: int | None = None

    current: _ClauseBuffer | None = None
    section_path = ""
    law_no = -1
    last_top_level_num: int | None = None
    last_appendix_letter: str | None = None
    in_appendix = False

    def _is_heading_like(text: str) -> bool:
        return bool(
            APPENDIX_HEADING_RE.match(text)
            or APPENDIX_LETTER_ONLY_RE.match(text)
            or LAW_HEADING_RE.match(text)
            or BARE_SECTION_INLINE_RE.match(text)
            or CLAUSE_NUMERIC_RE.match(text)
            or CLAUSE_APPENDIX_RE.match(text)
        )

    i = 0
    n = len(stream)
    while i < n:
        page_num, line = stream[i]

        # --- Appendix heading: inline "APPENDIX C LAWS 6 (THE PITCH)", or a
        # bare letter whose title wraps onto the next 1-2 lines ---
        m = APPENDIX_HEADING_RE.match(line)
        candidate_letter: str | None = None
        candidate_title = ""
        consumed = 1
        if m:
            candidate_letter = m.group(1).upper()
            candidate_title = m.group(2).strip()
        else:
            m2 = APPENDIX_LETTER_ONLY_RE.match(line)
            if m2:
                candidate_letter = m2.group(1).upper()
                # Prefer a run of ALL-CAPS continuation lines (MCC's style);
                # otherwise fall back to a single short line of any case
                # (IPL's style), so we don't sweep up the first clause's text.
                title_parts: list[str] = []
                j = i + 1
                while j < n and len(title_parts) < APPENDIX_TITLE_LOOKAHEAD:
                    cand = stream[j][1]
                    if _is_heading_like(cand) or not is_titleish_upper(cand):
                        break
                    title_parts.append(cand)
                    j += 1
                if not title_parts and j < n:
                    cand = stream[j][1]
                    if not _is_heading_like(cand) and 2 <= len(cand) <= 40:
                        title_parts.append(cand)
                        j += 1
                candidate_title = " ".join(title_parts)
                consumed = j - i

        if candidate_letter is not None:
            title_ok = len(candidate_title) >= MIN_APPENDIX_TITLE_LEN
            letter_ok = last_appendix_letter is None or candidate_letter > last_appendix_letter
            # A table of contents lists the appendices too, sometimes with
            # the same multi-line-wrapped-title shape as a real heading (MCC
            # wraps both in a similar way, just in Title Case up front vs.
            # ALL-CAPS in the body). Appendices only make sense after the
            # main numbered content has actually started, so require that
            # first -- this rejects the front-of-book ToC without needing to
            # rely on casing, which isn't consistent across documents.
            started_ok = last_top_level_num is not None
            if title_ok and letter_ok and started_ok:
                _flush(current, stream[i - 1][0] if i > 0 else page_num, records)
                current = None
                in_appendix = True
                last_appendix_letter = candidate_letter
                law_no = -1
                section_path = f"APPENDIX {candidate_letter} {candidate_title}"
                i += consumed
                continue

        # --- MCC-style law heading: "LAW 32 CAUGHT" ---
        m = LAW_HEADING_RE.match(line)
        if m and is_titleish_upper(m.group(2)):
            candidate = int(m.group(1))
            if last_top_level_num is None or candidate > last_top_level_num:
                _flush(current, stream[i - 1][0] if i > 0 else page_num, records)
                current = None
                in_appendix = False
                law_no = candidate
                last_top_level_num = candidate
                title = m.group(2).strip()
                section_path = f"LAW {law_no} {title}"
                i += 1
                continue

        # --- ICC/IPL-style bare section heading: "38 RUN OUT" (or split across lines) ---
        m = BARE_SECTION_INLINE_RE.match(line)
        if m:
            candidate = int(m.group(1))
            still_increasing = last_top_level_num is None or candidate > last_top_level_num
            if still_increasing and is_titleish_upper(m.group(2)):
                _flush(current, stream[i - 1][0] if i > 0 else page_num, records)
                current = None
                in_appendix = False
                law_no = candidate
                last_top_level_num = candidate
                title = m.group(2).strip()
                section_path = f"{law_no} {title}"
                i += 1
                continue
        m = BARE_SECTION_NUMBER_ONLY_RE.match(line)
        if m and i + 1 < n and stream[i + 1][0] in (page_num, page_num + 1):
            candidate = int(m.group(1))
            still_increasing = last_top_level_num is None or candidate > last_top_level_num
            next_line = stream[i + 1][1]
            if still_increasing and is_titleish_upper(next_line) and len(next_line) <= 80:
                _flush(current, stream[i - 1][0] if i > 0 else page_num, records)
                current = None
                in_appendix = False
                law_no = candidate
                last_top_level_num = candidate
                section_path = f"{law_no} {next_line}"
                i += 2
                continue

        # --- Clause heading: exactly one decimal point ---
        # Appendices normally use their own letter-prefixed numbering
        # ("A.1"), but IPL and ICC re-use plain "1.1"-style numbers inside
        # appendices too -- prefix those with the appendix letter so they
        # don't collide with the main body's clause of the same number.
        m = CLAUSE_APPENDIX_RE.match(line) if in_appendix else None
        numeric_under_appendix = False
        if m is None:
            m = CLAUSE_NUMERIC_RE.match(line)
            numeric_under_appendix = in_appendix and m is not None
        if m:
            clause_no = m.group(1)
            if numeric_under_appendix:
                clause_no = f"{last_appendix_letter}-{clause_no}"
            inline_title = (m.group(2) or "").strip()
            consumed = 1
            title = inline_title
            if not title and i + 1 < n and stream[i + 1][0] in (page_num, page_num + 1):
                candidate_title = stream[i + 1][1]
                looks_like_number = bool(
                    CLAUSE_NUMERIC_RE.match(candidate_title)
                    or CLAUSE_APPENDIX_RE.match(candidate_title)
                    or SUBCLAUSE_NUMERIC_RE.match(candidate_title)
                    or SUBCLAUSE_APPENDIX_RE.match(candidate_title)
                )
                if not looks_like_number and len(candidate_title) <= 80:
                    title = candidate_title
                    consumed = 2
            _flush(current, stream[i - 1][0] if i > 0 else page_num, records)
            current = _ClauseBuffer(
                law_no=law_no,
                clause=clause_no,
                clause_title=title,
                section_path=f"{section_path} > {clause_no}" if section_path else clause_no,
                page_start=page_num,
            )
            i += consumed
            continue

        # --- Sub-clause: folds into the current clause, no new chunk ---
        if current is not None and (SUBCLAUSE_NUMERIC_RE.match(line) or SUBCLAUSE_APPENDIX_RE.match(line)):
            current.lines.append(line)
            i += 1
            continue

        # --- Ordinary body text ---
        if current is not None:
            current.lines.append(line)
        else:
            if preamble_page_start is None:
                preamble_page_start = page_num
            preamble_lines.append(line)
        i += 1

    last_page = stream[-1][0] if stream else 1
    _flush(current, last_page, records)

    if preamble_lines:
        records.insert(
            0,
            {
                "law_no": -1,
                "clause": "preamble",
                "clause_title": "Preamble",
                "section_path": "Preamble",
                "page_start": preamble_page_start or 1,
                "page_end": records[0]["page_start"] if records else last_page,
                "text": "\n".join(preamble_lines),
            },
        )

    return _dedupe_by_clause(records)


def _dedupe_by_clause(records: list[dict]) -> list[dict]:
    """Keep the longest record for each clause number.

    A back-of-book "quick index" page (seen in MCC, pages 87-88) can re-list
    a clause number with just its one-line title and no real body; if that
    title happens to wrap onto a second line, the wrapped line looks like
    body text and produces a second, spurious record for a clause that
    already has real content earlier in the document. Keeping the longest
    text for each clause number discards the spurious short one.
    """
    best: dict[str, dict] = {}
    order: list[str] = []
    for record in records:
        key = record["clause"]
        if key not in best:
            order.append(key)
            best[key] = record
        elif len(record["text"]) > len(best[key]["text"]):
            best[key] = record
    return [best[key] for key in order]


def build_chunks(doc_id: str, source: dict, clause_records: list[dict]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[Chunk] = []

    for record in clause_records:
        pieces = splitter.split_text(record["text"])
        for chunk_index, piece in enumerate(pieces):
            header = f"[{source['title']} | Clause {record['clause']} {record['clause_title']}]".rstrip()
            header = re.sub(r"\s+\]$", "]", header)
            text = f"{header}\n{piece}"
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{record['clause']}::{chunk_index}",
                    doc_id=doc_id,
                    body=source["body"],
                    competition=source["competition"],
                    gender=source.get("gender", "all"),
                    doc_type=source["doc_type"],
                    effective_from=str(source["effective_from"]),
                    law_no=record["law_no"],
                    clause=record["clause"],
                    clause_title=record["clause_title"],
                    section_path=record["section_path"],
                    page_start=record["page_start"],
                    page_end=record["page_end"],
                    chunk_index=chunk_index,
                    content_hash=content_hash,
                    status=source.get("status", "active"),
                    text=text,
                )
            )
    return chunks


def chunk_document(doc_id: str, source: dict) -> list[Chunk]:
    pages = load_pages(doc_id)
    clause_records = parse_clauses(pages)
    return build_chunks(doc_id, source, clause_records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_ids", nargs="*", help="doc_ids to chunk (default: every doc in sources.yaml)")
    args = parser.parse_args()

    configure_logging()
    sources = load_sources()
    doc_ids = args.doc_ids or list(sources.keys())

    all_chunks: list[Chunk] = []
    for doc_id in doc_ids:
        if doc_id not in sources:
            logger.error("%s: not found in sources.yaml", doc_id)
            return 1
        pages_path = PROCESSED_DIR / f"{doc_id}.pages.jsonl"
        if not pages_path.exists():
            logger.error("%s: %s not found; run 'python -m src.parse' first", doc_id, pages_path)
            return 1

        chunks = chunk_document(doc_id, sources[doc_id])
        n_clauses = len({c.clause for c in chunks})
        lengths = [len(c.text) for c in chunks]
        logger.info(
            "%s: %d clauses -> %d chunks (avg %.0f chars, max %d chars)",
            doc_id,
            n_clauses,
            len(chunks),
            sum(lengths) / len(lengths) if lengths else 0,
            max(lengths) if lengths else 0,
        )
        all_chunks.extend(chunks)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

    logger.info("Wrote %d total chunks -> %s", len(all_chunks), CHUNKS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
