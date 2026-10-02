"""All tunable values for ThirdUmpire, overridable via environment variables.

No magic numbers should live in other modules -- add them here instead.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Embeddings ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Set to "Represent this sentence for searching relevant passages: " to test.
BGE_QUERY_INSTRUCTION = os.getenv("BGE_QUERY_INSTRUCTION", "")

# --- Reranker ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_FT_PATH = os.getenv("RERANKER_FT_PATH", "models/reranker-ft")

# --- Retrieval / chunking ---
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "20"))
CONTEXT_K = int(os.getenv("CONTEXT_K", "5"))
# For comparison questions: extra candidates fetched per compared competition
# that the main retrieval pass returned none of.
COMPARISON_TOPUP_K = int(os.getenv("COMPARISON_TOPUP_K", "3"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Hybrid search (Phase 5) ---
RRF_K = int(os.getenv("RRF_K", "60"))
# Decision (PROJECT_PLAN.md Phase 5), from eval/results/20261001T035703Z.json
# on the 48-question golden set: dense Recall@20=0.977 vs hybrid=0.953, and
# dense is faster (p50 0.066s vs 0.116s). The gap is entirely the "conduct"
# question type (IPL Code of Conduct articles): BM25 actively hurts there,
# since neighbouring articles share almost identical generic wording
# ("Note: Article N.N covers...") and lexically outscore the right one.
DEFAULT_FIRST_STAGE = os.getenv("DEFAULT_FIRST_STAGE", "dense")

# --- Reranker fine-tuning (Phase 6) ---
SYNTHETIC_QUESTIONS_MIN = int(os.getenv("SYNTHETIC_QUESTIONS_MIN", "2"))
SYNTHETIC_QUESTIONS_MAX = int(os.getenv("SYNTHETIC_QUESTIONS_MAX", "3"))
# Gemini free tier caps gemini-3.6-flash at 20 requests/DAY (confirmed by a
# live 429 -- GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue
# 20), not just a per-minute rate limit. Batched high (not the plan's
# original 5) specifically to make each of those 20 requests count: at ~25
# chunks/request, one day's quota covers ~500 chunks instead of ~100.
SYNTHETIC_CHUNKS_PER_REQUEST = int(os.getenv("SYNTHETIC_CHUNKS_PER_REQUEST", "25"))
SYNTHETIC_REQUEST_DELAY_SECONDS = float(os.getenv("SYNTHETIC_REQUEST_DELAY_SECONDS", "1.0"))
# Default run is a representative subset, not the full corpus (~6,000 chunks
# would be 1,000+ Gemini calls) -- see PROJECT_PLAN.md Phase 6. Pass
# --full to train/gen_synthetic.py to override.
SYNTHETIC_SUBSET_SIZE = int(os.getenv("SYNTHETIC_SUBSET_SIZE", "600"))
SYNTHETIC_SUBSET_SEED = int(os.getenv("SYNTHETIC_SUBSET_SEED", "42"))

LEAKAGE_SIMILARITY_THRESHOLD = float(os.getenv("LEAKAGE_SIMILARITY_THRESHOLD", "0.85"))

NEGATIVE_MINING_TOP_K = int(os.getenv("NEGATIVE_MINING_TOP_K", "30"))
MAX_NEGATIVES_PER_POSITIVE = int(os.getenv("MAX_NEGATIVES_PER_POSITIVE", "4"))
FALSE_NEGATIVE_SIMILARITY_THRESHOLD = float(os.getenv("FALSE_NEGATIVE_SIMILARITY_THRESHOLD", "0.97"))

TRAIN_VAL_SPLIT = float(os.getenv("TRAIN_VAL_SPLIT", "0.9"))

RERANKER_FT_EPOCHS = int(os.getenv("RERANKER_FT_EPOCHS", "2"))
RERANKER_FT_LR = float(os.getenv("RERANKER_FT_LR", "2e-5"))
RERANKER_FT_BATCH_SIZE = int(os.getenv("RERANKER_FT_BATCH_SIZE", "16"))
RERANKER_FT_WARMUP_RATIO = float(os.getenv("RERANKER_FT_WARMUP_RATIO", "0.1"))
RERANKER_FT_MAX_LENGTH = int(os.getenv("RERANKER_FT_MAX_LENGTH", "512"))

# --- Question / prompt ---
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3")

# --- Gemini ---
# No silent default: callers must go through require_gemini_model() so a
# missing value fails with a clear message at the point of use, not on import.
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
# Optional separate model for train/gen_synthetic.py, so Phase 6 data
# generation doesn't share GEMINI_MODEL's daily free-tier quota (tracked
# per-model) with production answering. Falls back to GEMINI_MODEL if unset.
SYNTHETIC_GEMINI_MODEL = os.getenv("SYNTHETIC_GEMINI_MODEL") or None
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))


def require_gemini_model() -> str:
    """Return GEMINI_MODEL, raising a clear error if it hasn't been set."""
    if not GEMINI_MODEL:
        raise RuntimeError(
            "GEMINI_MODEL is not set. Add it to your .env file, e.g. "
            "GEMINI_MODEL=gemini-2.5-flash (verify the current Flash model "
            "name in Google AI Studio)."
        )
    return GEMINI_MODEL


def require_synthetic_gemini_model() -> str:
    """Return SYNTHETIC_GEMINI_MODEL, falling back to GEMINI_MODEL (and its
    same not-set error) when no separate model is configured."""
    return SYNTHETIC_GEMINI_MODEL or require_gemini_model()


# --- Chroma ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = os.getenv("COLLECTION", "cricket_rules")

# --- LangSmith tracing ---
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "thirdumpire")


def configure_logging(level: int = logging.INFO) -> None:
    """Basic logging setup shared by all CLI entry points."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
