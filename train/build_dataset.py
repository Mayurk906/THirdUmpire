"""Final (query, passage, label) training data for reranker fine-tuning
(PROJECT_PLAN.md Phase 6).

1. Leakage guard: drops any synthetic question whose embedding cosine
   similarity to any golden question (single-turn and multi-turn, when that
   file exists) is >= LEAKAGE_SIMILARITY_THRESHOLD -- the golden sets are
   never used in training or validation.
2. Splits the remaining questions into train/val by (doc_id, clause) group
   (TRAIN_VAL_SPLIT), so no clause's text appears in both splits.
3. Writes one positive row (label=1) and one row per mined negative
   (label=0) per question to train/data/train.jsonl and val.jsonl.
4. Records INDEX_VERSION and DEFAULT_FIRST_STAGE (the first stage the
   negatives were mined with) in train/data/dataset_info.json.

Usage: python -m train.build_dataset
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from src.config import (
    BASE_DIR,
    DEFAULT_FIRST_STAGE,
    LEAKAGE_SIMILARITY_THRESHOLD,
    TRAIN_VAL_SPLIT,
    configure_logging,
)
from src.schemas import Chunk
from train.gen_synthetic import SyntheticQuestion
from train.mine_negatives import MinedNegatives

logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "train" / "data"
SYNTHETIC_PATH = DATA_DIR / "synthetic.jsonl"
NEGATIVES_PATH = DATA_DIR / "negatives.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"

GOLDEN_PATH = BASE_DIR / "eval" / "golden.jsonl"
GOLDEN_MULTITURN_PATH = BASE_DIR / "eval" / "golden_multiturn.jsonl"

SPLIT_SEED = 42


def load_synthetic(path: Path = SYNTHETIC_PATH) -> list[SyntheticQuestion]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(SyntheticQuestion(**json.loads(line)))
    return items


def load_negatives(path: Path = NEGATIVES_PATH) -> dict[str, list[str]]:
    by_qid: dict[str, list[str]] = {}
    if not path.exists():
        return by_qid
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = MinedNegatives(**json.loads(line))
                by_qid[row.qid] = row.negative_chunk_ids
    return by_qid


def load_golden_questions(path: Path) -> list[str]:
    if not path.exists():
        return []
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line)["question"])
    return questions


def compute_leaked_qids(
    synthetic: list[SyntheticQuestion],
    golden_questions: list[str],
    embed_fn,
    threshold: float = LEAKAGE_SIMILARITY_THRESHOLD,
) -> set[str]:
    """Pure core of the leakage guard: `embed_fn(texts) -> list[list[float]]`
    is injected (normalized embeddings assumed, so dot product = cosine) so
    this is testable without loading the real embedding model."""
    if not synthetic or not golden_questions:
        return set()

    synthetic_embeddings = embed_fn([sq.question for sq in synthetic])
    golden_embeddings = embed_fn(golden_questions)

    leaked: set[str] = set()
    for sq, s_emb in zip(synthetic, synthetic_embeddings):
        for g_emb in golden_embeddings:
            similarity = sum(a * b for a, b in zip(s_emb, g_emb))
            if similarity >= threshold:
                leaked.add(sq.qid)
                break
    return leaked


def group_key(sq: SyntheticQuestion) -> tuple[str, str]:
    return (sq.doc_id, sq.clause)


def split_groups(group_keys: list[tuple[str, str]], split: float = TRAIN_VAL_SPLIT, seed: int = SPLIT_SEED) -> tuple[set, set]:
    """Deterministically splits unique group keys into (train_keys, val_keys)."""
    unique = sorted(set(group_keys))
    rng = random.Random(seed)
    rng.shuffle(unique)
    cut = round(len(unique) * split)
    return set(unique[:cut]), set(unique[cut:])


def build_rows(sq: SyntheticQuestion, negative_chunk_ids: list[str], chunks_by_id: dict[str, Chunk]) -> list[dict]:
    rows: list[dict] = []
    positive = chunks_by_id.get(sq.pos_chunk_id)
    if positive is None:
        logger.warning("%s: positive chunk %s not found, skipping", sq.qid, sq.pos_chunk_id)
        return rows
    rows.append({"query": sq.question, "passage": positive.text, "label": 1})
    for neg_id in negative_chunk_ids:
        negative = chunks_by_id.get(neg_id)
        if negative is None:
            continue
        rows.append({"query": sq.question, "passage": negative.text, "label": 0})
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _index_version() -> str:
    from src.config import CHROMA_DIR

    path = BASE_DIR / CHROMA_DIR / "index_version.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else "unknown"


def main() -> int:
    configure_logging()

    if not SYNTHETIC_PATH.exists():
        logger.error("%s not found; run 'python -m train.gen_synthetic' first", SYNTHETIC_PATH)
        return 1

    synthetic = load_synthetic()
    negatives_by_qid = load_negatives()
    logger.info("Loaded %d synthetic questions, %d with mined negatives", len(synthetic), len(negatives_by_qid))

    golden_questions = load_golden_questions(GOLDEN_PATH) + load_golden_questions(GOLDEN_MULTITURN_PATH)
    logger.info("Checking leakage against %d golden question(s)", len(golden_questions))

    from src.index import get_embedder, load_chunks

    embedder = get_embedder()
    leaked_qids = compute_leaked_qids(synthetic, golden_questions, embedder.embed_documents)
    logger.info("Dropped %d/%d synthetic question(s) for leakage (similarity >= %.2f)", len(leaked_qids), len(synthetic), LEAKAGE_SIMILARITY_THRESHOLD)

    kept = [sq for sq in synthetic if sq.qid not in leaked_qids]

    train_keys, val_keys = split_groups([group_key(sq) for sq in kept])
    logger.info("Split %d (doc_id, clause) groups -> %d train, %d val", len(train_keys) + len(val_keys), len(train_keys), len(val_keys))

    chunks_by_id = {c.chunk_id: c for c in load_chunks()}

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    train_questions = 0
    val_questions = 0
    for sq in kept:
        key = group_key(sq)
        rows = build_rows(sq, negatives_by_qid.get(sq.qid, []), chunks_by_id)
        if not rows:
            continue
        if key in train_keys:
            train_rows.extend(rows)
            train_questions += 1
        else:
            val_rows.extend(rows)
            val_questions += 1

    write_jsonl(train_rows, TRAIN_PATH)
    write_jsonl(val_rows, VAL_PATH)

    info = {
        "index_version": _index_version(),
        "default_first_stage": DEFAULT_FIRST_STAGE,
        "synthetic_total": len(synthetic),
        "leaked_dropped": len(leaked_qids),
        "kept_questions": len(kept),
        "train_questions": train_questions,
        "val_questions": val_questions,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }
    DATASET_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")

    logger.info(
        "Wrote %d train rows (%d questions) -> %s, %d val rows (%d questions) -> %s",
        len(train_rows), train_questions, TRAIN_PATH, len(val_rows), val_questions, VAL_PATH,
    )
    logger.info("Dataset info -> %s", DATASET_INFO_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
