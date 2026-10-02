"""Validate every citation the model returns against the context actually passed to it."""
from __future__ import annotations

from src.schemas import Answer, Chunk, Citation


def find_matching_chunk(citation: Citation, context_chunks: list[Chunk]) -> Chunk | None:
    """The context chunk a citation refers to, or None if it doesn't match
    any of them. A citation matches if (doc_id, clause) is identical -- or
    the citation's clause is a more specific sub-clause of one (e.g. a chunk
    for clause "28.3" also covers "28.3.2", since sub-clauses are folded
    into their parent's chunk text; see chunk.py). Used both to validate
    citations and (in app.py) to show a citation's actual source text."""
    for c in context_chunks:
        if c.doc_id == citation.doc_id and (citation.clause == c.clause or citation.clause.startswith(f"{c.clause}.")):
            return c
    return None


def validate_citations(answer: Answer, context_chunks: list[Chunk]) -> tuple[bool, list[Citation]]:
    """Returns (citation_valid, invalid_citations).

    A citation is valid if it matches a context chunk (see
    find_matching_chunk) and page lies within that chunk's
    page_start-page_end. Invalid citations are reported, never silently
    dropped from answer.citations. If found=true but there are zero
    citations, citation_valid is False.
    """
    invalid: list[Citation] = []
    for citation in answer.citations:
        chunk = find_matching_chunk(citation, context_chunks)
        if chunk is None or not (chunk.page_start <= citation.page <= chunk.page_end):
            invalid.append(citation)

    if answer.found and not answer.citations:
        return False, invalid
    return len(invalid) == 0, invalid
