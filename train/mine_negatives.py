"""Hard-negative mining for reranker fine-tuning (PROJECT_PLAN.md Phase 6).

For each synthetic question, retrieves the top NEGATIVE_MINING_TOP_K with
DEFAULT_FIRST_STAGE *without* a competition filter, so same-clause chunks
from other competitions are reachable as candidates. A candidate is a
negative if its (doc_id, clause) differs from the question's positive.

False-negative guard: if the question does not name a specific competition,
a candidate whose chunk text is near-duplicate (cosine similarity >= 0.97)
of the positive's text is dropped -- it likely contains the same answer
under a different competition and would be a false negative. If the
question *does* name a competition, same-clause chunks from other
competitions are kept even when textually similar, since their differing
context header is exactly the hard case the reranker needs to learn from.

Usage: python -m train.mine_negatives
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import (
    BASE_DIR,
    DEFAULT_FIRST_STAGE,
    FALSE_NEGATIVE_SIMILARITY_THRESHOLD,
    MAX_NEGATIVES_PER_POSITIVE,
    NEGATIVE_MINING_TOP_K,
    configure_logging,
)
from src.schemas import Chunk, QueryPlan, ScoredChunk
from train.gen_synthetic import SyntheticQuestion, competition_label

logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "train" / "data"
SYNTHETIC_PATH = DATA_DIR / "synthetic.jsonl"
OUTPUT_PATH = DATA_DIR / "negatives.jsonl"


class MinedNegatives(BaseModel):
    qid: str
    negative_chunk_ids: list[str] = Field(default_factory=list)


def load_synthetic(path: Path = SYNTHETIC_PATH) -> list[SyntheticQuestion]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(SyntheticQuestion(**json.loads(line)))
    return items


def is_positive(chunk: ScoredChunk, sq: SyntheticQuestion) -> bool:
    return chunk.chunk.doc_id == sq.doc_id and chunk.chunk.clause == sq.clause


def filter_candidates(
    sq: SyntheticQuestion,
    retrieved: list[ScoredChunk],
    positive_text: str,
    similarity_fn,
) -> list[ScoredChunk]:
    """Pure core of the mining logic: `retrieved` is the already-fetched
    top-k (no competition filter), `similarity_fn(text_a, text_b) -> float`
    is injected so this is testable without an embedder."""
    candidates = [sc for sc in retrieved if not is_positive(sc, sq)]

    if competition_label(sq.competition) is None:
        candidates = [sc for sc in candidates if similarity_fn(positive_text, sc.chunk.text) < FALSE_NEGATIVE_SIMILARITY_THRESHOLD]

    return candidates[:MAX_NEGATIVES_PER_POSITIVE]


def _cosine_similarity_fn(embedder):
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def similarity(text_a: str, text_b: str) -> float:
        emb_a, emb_b = embedder.embed_documents([text_a, text_b])
        return _dot(emb_a, emb_b)

    return similarity


def mine_for_question(
    sq: SyntheticQuestion,
    chunks_by_id: dict[str, Chunk],
    similarity_fn,
    first_stage: str = DEFAULT_FIRST_STAGE,
) -> MinedNegatives:
    from src.retrieve import retrieve

    plan = QueryPlan(question=sq.question, competition=None, allowed_competitions=None)
    retrieved = retrieve(plan, k=NEGATIVE_MINING_TOP_K, first_stage=first_stage)

    positive_chunk = chunks_by_id.get(sq.pos_chunk_id)
    positive_text = positive_chunk.text if positive_chunk else ""

    kept = filter_candidates(sq, retrieved, positive_text, similarity_fn)
    return MinedNegatives(qid=sq.qid, negative_chunk_ids=[sc.chunk.chunk_id for sc in kept])


def load_done_qids(path: Path = OUTPUT_PATH) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["qid"])
    return done


def append_result(result: MinedNegatives, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(result.model_dump_json() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-stage", choices=["dense", "hybrid"], default=DEFAULT_FIRST_STAGE)
    args = parser.parse_args()

    configure_logging()

    if not SYNTHETIC_PATH.exists():
        logger.error("%s not found; run 'python -m train.gen_synthetic' first", SYNTHETIC_PATH)
        return 1

    questions = load_synthetic()
    done_qids = load_done_qids()
    remaining = [q for q in questions if q.qid not in done_qids]
    logger.info("Target: %d questions (%d already done, %d remaining)", len(questions), len(questions) - len(remaining), len(remaining))
    if not remaining:
        logger.info("Nothing to do -- all questions already have mined negatives.")
        return 0

    from src.index import get_embedder, load_chunks

    chunks_by_id = {c.chunk_id: c for c in load_chunks()}
    embedder = get_embedder()
    similarity_fn = _cosine_similarity_fn(embedder)

    for i, sq in enumerate(remaining, start=1):
        result = mine_for_question(sq, chunks_by_id, similarity_fn, first_stage=args.first_stage)
        append_result(result)
        if i % 50 == 0 or i == len(remaining):
            logger.info("%d/%d questions mined", i, len(remaining))

    logger.info("Done. Negatives written to %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
