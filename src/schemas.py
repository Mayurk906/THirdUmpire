"""Pydantic models shared across modules."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Competitions known to the corpus today.
KNOWN_COMPETITIONS = (
    "laws",
    "ipl",
    "icc_t20i",
    "icc_test",
    "icc_odi",
    "bcci_domestic_multiday",
    "bcci_domestic_odi",
    "bcci_domestic_t20",
)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    body: str
    competition: str
    gender: str
    doc_type: str
    effective_from: str
    law_no: int
    clause: str
    clause_title: str
    section_path: str
    page_start: int
    page_end: int
    chunk_index: int
    content_hash: str
    status: str
    text: str

    def metadata(self) -> dict[str, str | int | float | bool]:
        """The Chroma metadata dict -- everything except the document text itself."""
        return self.model_dump(exclude={"text"})


class QueryPlan(BaseModel):
    question: str
    # The competition context after applying any explicit override:
    # "ipl" | "icc_t20i" | "laws" | None (no competition named -> no filter).
    competition: str | None
    # None means no competition filter (search every competition); otherwise
    # the list of competition values retrieval must restrict to.
    allowed_competitions: list[str] | None
    # Set when the question compares two or more competitions; the pipeline
    # guarantees each one at least one slot in the answer's context.
    compared_competitions: list[str] | None = None


class ScoredChunk(BaseModel):
    """A chunk plus a retrieval or reranking score, threaded through the pipeline.

    `score` is the primary ranking score (cosine similarity for retrieval,
    sigmoid-normalised relevance in [0, 1] for reranking). `raw_score` is
    only set after reranking, holding the cross-encoder's raw logit.
    """

    chunk: Chunk
    score: float
    raw_score: float | None = None


class Citation(BaseModel):
    doc_id: str
    clause: str
    page: int


class Answer(BaseModel):
    answer: str
    found: bool
    applies_to: Literal[
        "laws",
        "ipl",
        "icc_t20i",
        "icc_test",
        "icc_odi",
        "bcci_domestic_multiday",
        "bcci_domestic_odi",
        "bcci_domestic_t20",
        "unknown",
    ]
    citations: list[Citation] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: Answer
    query_plan: QueryPlan
    citation_valid: bool
    invalid_citations: list[Citation] = Field(default_factory=list)
    latencies: dict[str, float] = Field(default_factory=dict)
    # The exact chunks passed to the LLM as context -- always populated
    # (unlike retrieved/reranked below) so a UI can show a citation's actual
    # source text without needing debug=True.
    context_chunks: list[Chunk] = Field(default_factory=list)
    retrieved: list[ScoredChunk] | None = None
    reranked: list[ScoredChunk] | None = None
