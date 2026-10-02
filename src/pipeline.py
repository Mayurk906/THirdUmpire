"""run_query(): the ONE entry point used by the CLI, Streamlit, and (Phase 9) FastAPI.

No retrieval or prompt logic belongs in ask.py, app.py, or api/ -- they all
call run_query() and just render the AskResponse.
"""
from __future__ import annotations

import re
import time

from src.answer import QuestionTooLongError, generate_answer
from src.citations import validate_citations
from src.config import COMPARISON_TOPUP_K, CONTEXT_K, DEFAULT_FIRST_STAGE, MAX_QUESTION_CHARS, RETRIEVE_K
from src.query_analyzer import build_query_plan
from src.rerank import rerank
from src.retrieve import retrieve
from src.schemas import Answer, AskResponse, Chunk, QueryPlan, ScoredChunk

_WOMEN_RE = re.compile(r"\bwom[ae]n'?s?\b", re.IGNORECASE)


def top_up_compared(plan: QueryPlan, retrieved: list[ScoredChunk], first_stage: str) -> list[ScoredChunk]:
    """Adds a few candidates for each compared competition the main pass
    returned none of -- with ~one near-identical clause per competition and
    gender in the corpus, a single top-k can be all T20 variants."""
    present = {sc.chunk.competition for sc in retrieved}
    seen = {sc.chunk.chunk_id for sc in retrieved}
    extra: list[ScoredChunk] = []
    for competition in plan.compared_competitions or []:
        if competition in present:
            continue
        sub_plan = QueryPlan(question=plan.question, competition=competition, allowed_competitions=[competition])
        for sc in retrieve(sub_plan, k=COMPARISON_TOPUP_K, first_stage=first_stage):
            if sc.chunk.chunk_id not in seen:
                seen.add(sc.chunk.chunk_id)
                extra.append(sc)
    return retrieved + extra


def select_context(
    reranked: list[ScoredChunk],
    k: int,
    compared_competitions: list[str] | None = None,
    question: str = "",
) -> list[Chunk]:
    """Top-k by rerank score, except that for a comparison question the
    best-ranked chunk of each compared competition is guaranteed a slot.
    Those guaranteed picks prefer the gender the question asks about
    (women's only if it says so, otherwise men's/all, matching the usual
    reading of an unqualified "Test" or "ODI")."""
    if not compared_competitions:
        return [sc.chunk for sc in reranked[:k]]

    wants_women = bool(_WOMEN_RE.search(question))

    def gender_ok(chunk: Chunk) -> bool:
        return (chunk.gender == "women") == wants_women

    chosen: list[int] = []
    for competition in compared_competitions:
        candidates = [i for i, sc in enumerate(reranked) if sc.chunk.competition == competition]
        preferred = [i for i in candidates if gender_ok(reranked[i].chunk)]
        pick = (preferred or candidates or [None])[0]
        if pick is not None and pick not in chosen:
            chosen.append(pick)
    chosen = sorted(chosen)[:k]

    for i in range(len(reranked)):
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    return [reranked[i].chunk for i in sorted(chosen)]


def run_query(
    question: str,
    competition: str | None = None,
    first_stage: str = DEFAULT_FIRST_STAGE,
    reranker: str = "base",
    debug: bool = False,
) -> AskResponse:
    plan = build_query_plan(question, competition)

    if len(question) > MAX_QUESTION_CHARS:
        answer = Answer(
            answer=f"Question is over the {MAX_QUESTION_CHARS}-character limit; please shorten it.",
            found=False,
            applies_to="unknown",
            citations=[],
        )
        return AskResponse(answer=answer, query_plan=plan, citation_valid=True, invalid_citations=[], latencies={})

    t0 = time.perf_counter()
    retrieved = retrieve(plan, k=RETRIEVE_K, first_stage=first_stage)
    if plan.compared_competitions:
        retrieved = top_up_compared(plan, retrieved, first_stage)
    t1 = time.perf_counter()
    reranked = rerank(question, retrieved, reranker=reranker)
    t2 = time.perf_counter()

    context_chunks = select_context(reranked, CONTEXT_K, plan.compared_competitions, question)
    try:
        answer = generate_answer(question, context_chunks)
    except QuestionTooLongError as exc:
        answer = Answer(answer=str(exc), found=False, applies_to="unknown", citations=[])
    t3 = time.perf_counter()

    citation_valid, invalid_citations = validate_citations(answer, context_chunks)

    latencies = {"retrieve": t1 - t0, "rerank": t2 - t1, "answer": t3 - t2}

    return AskResponse(
        answer=answer,
        query_plan=plan,
        citation_valid=citation_valid,
        invalid_citations=invalid_citations,
        latencies=latencies,
        context_chunks=context_chunks,
        retrieved=retrieved if debug else None,
        reranked=reranked if debug else None,
    )
