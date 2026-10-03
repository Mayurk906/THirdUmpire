"""run_query(): the ONE entry point used by the CLI, Streamlit, and (Phase 9) FastAPI.

No retrieval or prompt logic belongs in ask.py, app.py, or api/ -- they all
call run_query() and just render the AskResponse.
"""
from __future__ import annotations

import re
import time

from src.answer import QuestionTooLongError, generate_answer
from src.answer_cache import cache_key
from src.answer_cache import get as cache_get
from src.answer_cache import put as cache_put
from src.citations import validate_citations
from src.config import (
    ANSWER_CACHE_ENABLED,
    COMPARISON_TOPUP_K,
    CONTEXT_K,
    DEFAULT_FIRST_STAGE,
    MAX_COMPETITIONS_PER_CLAUSE_TITLE,
    MAX_QUESTION_CHARS,
    QUERY_PLANNER_MODE,
    RETRIEVE_K,
)
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


def _title_key(clause_title: str) -> str:
    """Case-insensitive grouping key for a clause_title. Found live: ICC
    men's T20I/ODI documents title this clause "Minimum over rates" while
    every other competition and the Laws use "Minimum Over Rates" -- a
    case-sensitive key would silently treat those as two different clauses
    and let the cap double its effective limit on exactly the content it's
    meant to cap."""
    return clause_title.strip().lower()


def _fill_diverse(
    reranked: list[ScoredChunk],
    k: int,
    chosen: list[int],
    max_per_clause_title: int,
) -> list[int]:
    """Fills `chosen` up to k in rank order, capping how many DIFFERENT
    (competition, gender) pairs can share a clause_title -- the corpus has
    ~15 near-identical copies of some clauses (one per competition x
    gender), and without this cap the top CONTEXT_K can fill entirely with
    duplicates of one clause and crowd out everything else. Multiple chunks
    of the SAME (competition, gender) under one clause_title are never
    capped, since those are sequential pieces of one long clause, not
    duplicates. A candidate skipped only for crowding is deferred and used
    to fill any slots a stricter diverse pass couldn't (e.g. a question with
    only one real clause worth 5 chunks' answer)."""
    variants: dict[str, set[tuple[str, str]]] = {}
    for i in chosen:
        chunk = reranked[i].chunk
        variants.setdefault(_title_key(chunk.clause_title), set()).add((chunk.competition, chunk.gender))

    chosen_set = set(chosen)
    deferred: list[int] = []
    for i, sc in enumerate(reranked):
        if len(chosen) >= k:
            break
        if i in chosen_set:
            continue
        variant = (sc.chunk.competition, sc.chunk.gender)
        seen = variants.setdefault(_title_key(sc.chunk.clause_title), set())
        if variant not in seen and len(seen) >= max_per_clause_title:
            deferred.append(i)
            continue
        seen.add(variant)
        chosen.append(i)
        chosen_set.add(i)

    for i in deferred:
        if len(chosen) >= k:
            break
        chosen.append(i)

    return chosen


def select_context(
    reranked: list[ScoredChunk],
    k: int,
    compared_competitions: list[str] | None = None,
    question: str = "",
    gender: str | None = None,
) -> list[Chunk]:
    """The final k chunks sent to the model: diversity-capped by clause_title
    x (competition, gender) -- see _fill_diverse -- with the best-ranked
    chunk of each compared competition guaranteed a slot first for
    comparison questions. Guaranteed picks prefer the gender the question
    asks about: `gender` if the query planner determined one (see
    QueryPlan.gender), else regex-matched from the question text. Either
    way, women's only if asked, otherwise men's/all, matching the usual
    reading of an unqualified "Test" or "ODI"."""
    chosen: list[int] = []
    if compared_competitions:
        wants_women = gender == "women" if gender else bool(_WOMEN_RE.search(question))

        def gender_ok(chunk: Chunk) -> bool:
            return (chunk.gender == "women") == wants_women

        for competition in compared_competitions:
            candidates = [i for i, sc in enumerate(reranked) if sc.chunk.competition == competition]
            preferred = [i for i in candidates if gender_ok(reranked[i].chunk)]
            pick = (preferred or candidates or [None])[0]
            if pick is not None and pick not in chosen:
                chosen.append(pick)
        chosen = chosen[:k]

    chosen = _fill_diverse(reranked, k, chosen, MAX_COMPETITIONS_PER_CLAUSE_TITLE)
    return [reranked[i].chunk for i in sorted(chosen)]


def run_query(
    question: str,
    competition: str | None = None,
    first_stage: str = DEFAULT_FIRST_STAGE,
    reranker: str = "base",
    debug: bool = False,
) -> AskResponse:
    if QUERY_PLANNER_MODE == "llm":
        from src.query_planner import build_query_plan_llm

        plan = build_query_plan_llm(question, competition)
    else:
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

    context_chunks = select_context(reranked, CONTEXT_K, plan.compared_competitions, question, plan.gender)

    key = cache_key(question, context_chunks) if ANSWER_CACHE_ENABLED else None
    cached_answer = cache_get(key) if key else None
    answer_cached = cached_answer is not None
    if cached_answer is not None:
        answer = cached_answer
    else:
        try:
            answer = generate_answer(question, context_chunks)
            if key:
                cache_put(key, answer)
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
        answer_cached=answer_cached,
    )
