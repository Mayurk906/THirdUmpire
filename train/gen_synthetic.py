"""Synthetic training questions for reranker fine-tuning (PROJECT_PLAN.md Phase 6).

For each active chunk, asks Gemini for a few natural questions answerable
ONLY from that chunk, naming the competition explicitly when the chunk is
competition-specific. Batches several chunks per request, rate-limits,
caches results to train/data/synthetic.jsonl, and is resumable (skips
chunk_ids already present in that file).

Usage:
  python -m train.gen_synthetic                 # representative subset (default)
  python -m train.gen_synthetic --full           # every active chunk (~6,000 -- see config.py)
  python -m train.gen_synthetic --limit 20       # small smoke-test run
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.config import (
    BASE_DIR,
    SYNTHETIC_CHUNKS_PER_REQUEST,
    SYNTHETIC_QUESTIONS_MAX,
    SYNTHETIC_QUESTIONS_MIN,
    SYNTHETIC_REQUEST_DELAY_SECONDS,
    SYNTHETIC_SUBSET_SEED,
    SYNTHETIC_SUBSET_SIZE,
    TEMPERATURE,
    configure_logging,
    require_synthetic_gemini_model,
)
from src.schemas import Chunk

logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "train" / "data"
OUTPUT_PATH = DATA_DIR / "synthetic.jsonl"

# Human-readable phrase a question must use to name the competition
# explicitly (PROJECT_PLAN.md Phase 6 step 1); None (laws) means no
# competition needs naming -- a Laws question is the general case.
_COMPETITION_LABELS: dict[str, str | None] = {
    "laws": None,
    "ipl": "IPL",
    "icc_t20i": "a T20I",
    "icc_test": "a Test match",
    "icc_odi": "an ODI",
}

# Ranji/Vijay Hazare/Syed Mushtaq Ali Trophy are specifically the MEN'S BCCI
# domestic competitions. A fine-tuned reranker trained on an earlier version
# of this file -- which used these same names for the *women's* chunks too,
# since competition_label() only keyed on `competition` -- learned that
# either gender's document could answer a "Ranji Trophy" question, and
# measurably confused men's/women's documents on the real eval set as a
# result (6 of 7 dense+ft_rerank failures were exactly this). Generic,
# gender-neutral phrasing is used for women's chunks instead, rather than
# guessing the real women's tournament names, which aren't confidently known
# here and guessing wrong would just reintroduce the same bug differently.
_MENS_ONLY_TROPHY_LABELS: dict[str, str] = {
    "bcci_domestic_multiday": "the Ranji Trophy (BCCI men's domestic multi-day cricket)",
    "bcci_domestic_odi": "the Vijay Hazare Trophy (BCCI men's domestic one-day cricket)",
    "bcci_domestic_t20": "the Syed Mushtaq Ali Trophy (BCCI men's domestic T20 cricket)",
}
_GENERIC_BCCI_DOMESTIC_LABELS: dict[str, str] = {
    "bcci_domestic_multiday": "BCCI women's domestic multi-day cricket",
    "bcci_domestic_odi": "BCCI women's domestic one-day cricket",
    "bcci_domestic_t20": "BCCI women's domestic T20 cricket",
}

_RETRYABLE_STATUS_HINTS = ("429", "500", "502", "503", "504")


def competition_label(competition: str, gender: str = "all") -> str | None:
    if competition in _MENS_ONLY_TROPHY_LABELS:
        if gender == "women":
            return _GENERIC_BCCI_DOMESTIC_LABELS[competition]
        return _MENS_ONLY_TROPHY_LABELS[competition]
    return _COMPETITION_LABELS.get(competition)


class SyntheticQuestion(BaseModel):
    qid: str
    question: str
    pos_chunk_id: str
    doc_id: str
    clause: str
    competition: str


class _ChunkQuestions(BaseModel):
    chunk_index: int
    questions: list[str] = Field(default_factory=list)


class _BatchResponse(BaseModel):
    items: list[_ChunkQuestions]


def select_subset(chunks: list[Chunk], size: int, seed: int = SYNTHETIC_SUBSET_SEED) -> list[Chunk]:
    """Stratified sample across doc_id so every document gets roughly even
    representation, rather than exhausting the quota on whichever doc_id
    happens to come first in the corpus."""
    if size >= len(chunks):
        return list(chunks)

    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)

    rng = random.Random(seed)
    for doc_chunks in by_doc.values():
        rng.shuffle(doc_chunks)

    doc_ids = sorted(by_doc.keys())
    selected: list[Chunk] = []
    cursors = {doc_id: 0 for doc_id in doc_ids}
    while len(selected) < size:
        progressed = False
        for doc_id in doc_ids:
            if len(selected) >= size:
                break
            i = cursors[doc_id]
            if i < len(by_doc[doc_id]):
                selected.append(by_doc[doc_id][i])
                cursors[doc_id] = i + 1
                progressed = True
        if not progressed:
            break
    return selected


def build_batches(chunks: list[Chunk], batch_size: int = SYNTHETIC_CHUNKS_PER_REQUEST) -> list[list[Chunk]]:
    return [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]


def build_prompt(chunks: list[Chunk]) -> str:
    parts = [
        f"Write {SYNTHETIC_QUESTIONS_MIN}-{SYNTHETIC_QUESTIONS_MAX} natural questions for EACH of the "
        f"{len(chunks)} numbered chunks below, for fine-tuning a cricket-rules retrieval reranker.\n"
        "Rules for every question:\n"
        "- It must be answerable using ONLY the information in that one chunk -- not from outside "
        "knowledge, and not from any other chunk.\n"
        "- It must read like a real question a player, fan, or commentator would ask -- not a reworded "
        "copy of the clause title, and never a meta-reference like \"what does this clause say\".\n"
        "- If a chunk's competition note below says it is specific to a competition, every question for "
        "that chunk MUST explicitly name exactly that competition phrase (e.g. \"In IPL, ...\", \"In a "
        "T20I, ...\") -- never substitute a different competition's name. If there is no competition "
        "note, write a general question naming no specific competition.\n"
    ]
    for i, chunk in enumerate(chunks):
        label = competition_label(chunk.competition, chunk.gender)
        note = f"Competition: specific to {label}." if label else "Competition: general (Laws of Cricket)."
        parts.append(f"\n[Chunk {i}]\n{note}\n{chunk.text}")
    return "\n".join(parts)


def parse_batch_response(chunks: list[Chunk], response: _BatchResponse) -> list[SyntheticQuestion]:
    results: list[SyntheticQuestion] = []
    by_index = {item.chunk_index: item.questions for item in response.items}
    for i, chunk in enumerate(chunks):
        questions = by_index.get(i, [])
        for q_num, question in enumerate(questions):
            results.append(
                SyntheticQuestion(
                    qid=f"syn_{chunk.chunk_id}_{q_num}",
                    question=question,
                    pos_chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    clause=chunk.clause,
                    competition=chunk.competition,
                )
            )
    return results


def load_done_chunk_ids(path: Path = OUTPUT_PATH) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["pos_chunk_id"])
    return done


def append_results(results: list[SyntheticQuestion], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")


def _get_llm() -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=require_synthetic_gemini_model(), temperature=TEMPERATURE)


def generate_batch_with_backoff(llm: Any, chunks: list[Chunk], max_retries: int = 4) -> list[SyntheticQuestion]:
    structured_llm = llm.with_structured_output(_BatchResponse)
    prompt = build_prompt(chunks)
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            response = structured_llm.invoke(prompt)
            parsed = response if isinstance(response, _BatchResponse) else _BatchResponse.model_validate(response)
            return parse_batch_response(chunks, parsed)
        except Exception as exc:
            retryable = any(hint in str(exc) for hint in _RETRYABLE_STATUS_HINTS)
            if not retryable or attempt == max_retries:
                logger.error("Batch starting at %s failed (attempt %d): %s", chunks[0].chunk_id, attempt, exc)
                return []
            logger.warning("Batch call failed (%s), retrying in %.0fs ...", exc, delay)
            time.sleep(delay)
            delay *= 2
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="generate for every active chunk, not just a subset")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of chunks (overrides --full/subset size)")
    args = parser.parse_args()

    configure_logging()

    from src.index import load_chunks

    all_chunks = [c for c in load_chunks() if c.status == "active"]
    if args.limit is not None:
        target = select_subset(all_chunks, args.limit)
    elif args.full:
        target = all_chunks
    else:
        target = select_subset(all_chunks, SYNTHETIC_SUBSET_SIZE)

    done_ids = load_done_chunk_ids()
    remaining = [c for c in target if c.chunk_id not in done_ids]
    logger.info(
        "Target: %d chunks (%d already done, %d remaining)", len(target), len(target) - len(remaining), len(remaining)
    )
    if not remaining:
        logger.info("Nothing to do -- all target chunks already have cached synthetic questions.")
        return 0

    llm = _get_llm()
    batches = build_batches(remaining)
    total_generated = 0
    for i, batch in enumerate(batches, start=1):
        results = generate_batch_with_backoff(llm, batch)
        append_results(results)
        total_generated += len(results)
        logger.info("Batch %d/%d: %d chunks -> %d questions (running total %d)", i, len(batches), len(batch), len(results), total_generated)
        if i < len(batches):
            time.sleep(SYNTHETIC_REQUEST_DELAY_SECONDS)

    logger.info("Done. %d questions written to %s", total_generated, OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
