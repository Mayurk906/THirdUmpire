"""Baseline retrieval + answer evaluation harness (PROJECT_PLAN.md Phase 3).

Usage:
  python -m eval.run_eval
  python -m eval.run_eval --golden eval/golden.jsonl --configs dense,dense+base_rerank
  python -m eval.run_eval --with-answers
  python -m eval.run_eval --limit 10          # quick smoke run

Never writes eval/golden.jsonl -- that file is user-written only (see
PROJECT_PLAN.md section 0, rule 3). Run against eval/golden_template.jsonl
to sanity-check the harness itself; the real numbers need the hand-verified
golden set.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from eval.metrics import hit_rank, mean, percentile, reciprocal_rank
from src.config import BASE_DIR, CHROMA_DIR, PROMPT_VERSION, RERANKER_FT_PATH, RETRIEVE_K, configure_logging
from src.pipeline import run_query
from src.query_analyzer import build_query_plan
from src.rerank import rerank
from src.retrieve import retrieve

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_PATH = BASE_DIR / "eval" / "golden.jsonl"
RESULTS_DIR = BASE_DIR / "eval" / "results"
RECALL_KS = (1, 5, 20)
MRR_K = 10
FAILURE_TOP_N = 5


# --- Golden set -------------------------------------------------------------


class GoldRef(BaseModel):
    doc_id: str
    clause: str


class GoldenItem(BaseModel):
    id: str
    question: str
    competition: Literal[
        "laws",
        "ipl",
        "icc_t20i",
        "icc_test",
        "icc_odi",
        "bcci_domestic_multiday",
        "bcci_domestic_odi",
        "bcci_domestic_t20",
        "any",
    ]
    gold: list[GoldRef] = Field(default_factory=list)
    answer_short: str = ""
    type: Literal["lookup", "format_dependent", "changed_2026", "unanswerable", "conduct"]
    notes: str = ""

    def gold_pairs(self) -> set[tuple[str, str]]:
        return {(g.doc_id, g.clause) for g in self.gold}

    def competition_override(self) -> str | None:
        return None if self.competition == "any" else self.competition


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                items.append(GoldenItem(**json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid golden row: {exc}") from exc
    return items


# --- Configs ------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    name: str
    first_stage: str
    reranker: str | None  # None = retrieval-only, no rerank stage


def _hybrid_available() -> bool:
    try:
        import src.bm25  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _ft_reranker_available() -> bool:
    return (BASE_DIR / RERANKER_FT_PATH).exists()


def available_configs() -> tuple[dict[str, EvalConfig], list[str]]:
    """Returns ({name: EvalConfig} for configs runnable now, [skip messages])."""
    hybrid_ok = _hybrid_available()
    ft_ok = _ft_reranker_available()

    candidates: list[tuple[EvalConfig, bool, str]] = [
        (EvalConfig("dense", "dense", None), True, ""),
        (EvalConfig("dense+base_rerank", "dense", "base"), True, ""),
        (EvalConfig("hybrid", "hybrid", None), hybrid_ok, "hybrid first stage not implemented yet (Phase 5)"),
        (
            EvalConfig("hybrid+base_rerank", "hybrid", "base"),
            hybrid_ok,
            "hybrid first stage not implemented yet (Phase 5)",
        ),
        (
            EvalConfig("dense+ft_rerank", "dense", "ft"),
            ft_ok,
            f"fine-tuned reranker not found at {RERANKER_FT_PATH} (Phase 6)",
        ),
    ]
    available = {cfg.name: cfg for cfg, ok, _ in candidates if ok}
    skips = [f"{cfg.name}: skipped ({reason})" for cfg, ok, reason in candidates if not ok]
    return available, skips


# --- Per-query evaluation -----------------------------------------------------


def _ranked_pairs(scored_chunks) -> list[tuple[str, str]]:
    return [(sc.chunk.doc_id, sc.chunk.clause) for sc in scored_chunks]


def eval_retrieval(cfg: EvalConfig, item: GoldenItem) -> dict:
    """Runs retrieval (+ rerank if the config has one) for one question.

    Calls retrieve()/rerank() directly rather than run_query() so retrieval
    quality can be measured for configs run_query() doesn't support (e.g.
    "dense" has no reranker, but run_query() always reranks).
    """
    plan = build_query_plan(item.question, item.competition_override())

    t0 = time.perf_counter()
    retrieved = retrieve(plan, k=RETRIEVE_K, first_stage=cfg.first_stage)
    t1 = time.perf_counter()

    if cfg.reranker:
        ranked = rerank(item.question, retrieved, reranker=cfg.reranker)
        t2 = time.perf_counter()
        rerank_latency: float | None = t2 - t1
    else:
        ranked = retrieved
        rerank_latency = None

    pairs = _ranked_pairs(ranked)
    rank = hit_rank(pairs, item.gold_pairs())
    return {
        "rank": rank,
        "retrieve_latency": t1 - t0,
        "rerank_latency": rerank_latency,
        "top": pairs[:FAILURE_TOP_N],
    }


def eval_answer(cfg: EvalConfig, item: GoldenItem) -> dict | None:
    """Runs the full pipeline for one question. None if the config has no
    reranker, since run_query() requires one (production always reranks)."""
    if cfg.reranker is None:
        return None

    response = run_query(
        item.question,
        competition=item.competition_override(),
        first_stage=cfg.first_stage,
        reranker=cfg.reranker,
        debug=False,
    )
    cited_pairs = {(c.doc_id, c.clause) for c in response.answer.citations}
    return {
        "found": response.answer.found,
        "citation_valid": response.citation_valid,
        "any_gold_cited": bool(cited_pairs & item.gold_pairs()),
        "retrieve_latency": response.latencies.get("retrieve"),
        "rerank_latency": response.latencies.get("rerank"),
        "answer_latency": response.latencies.get("answer"),
    }


# --- Aggregation ---------------------------------------------------------------


def _retrieval_metrics(rows: list[dict]) -> dict:
    ranks = [r["rank"] for r in rows]
    n = len(ranks)
    metrics = {f"recall_at_{k}": (sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0) for k in RECALL_KS}
    metrics["mrr_at_10"] = mean([reciprocal_rank(r, MRR_K) for r in ranks]) if n else 0.0
    metrics["n"] = n
    return metrics


def _latency_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"p50": percentile(values, 50), "p95": percentile(values, 95), "mean": mean(values)}


def evaluate_config(cfg: EvalConfig, golden: list[GoldenItem], with_answers: bool) -> tuple[dict, dict[str, dict]]:
    """Returns (result, retrieval_rows) -- retrieval_rows keyed by golden id,
    reused by the caller for the failures CSV so retrieval only runs once
    per (config, item)."""
    answerable = [item for item in golden if item.type != "unanswerable"]
    unanswerable = [item for item in golden if item.type == "unanswerable"]

    retrieval_rows: dict[str, dict] = {}
    for item in golden:
        retrieval_rows[item.id] = eval_retrieval(cfg, item)

    overall = _retrieval_metrics([retrieval_rows[item.id] for item in answerable])
    by_type: dict[str, dict] = {}
    for t in sorted({item.type for item in answerable}):
        by_type[t] = _retrieval_metrics([retrieval_rows[item.id] for item in answerable if item.type == t])

    retrieve_latencies = [retrieval_rows[item.id]["retrieve_latency"] for item in golden]
    rerank_latencies = [
        retrieval_rows[item.id]["rerank_latency"] for item in golden if retrieval_rows[item.id]["rerank_latency"] is not None
    ]

    result = {
        "config": cfg.name,
        "first_stage": cfg.first_stage,
        "reranker": cfg.reranker,
        "n_questions": len(golden),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "retrieval": {"overall": overall, "by_type": by_type},
        "latency": {
            "retrieve": _latency_stats(retrieve_latencies),
            "rerank": _latency_stats(rerank_latencies),
        },
    }

    if with_answers:
        if cfg.reranker is None:
            result["answers"] = None
            result["answers_skip_reason"] = "run_query() requires a reranker; use a +*_rerank config"
        else:
            answer_rows = [eval_answer(cfg, item) for item in golden]
            found_rows = [r for r in answer_rows if r["found"]]
            answerable_rows = [r for item, r in zip(golden, answer_rows) if item.type != "unanswerable"]
            unanswerable_rows = [r for item, r in zip(golden, answer_rows) if item.type == "unanswerable"]

            result["answers"] = {
                "citation_accuracy": mean([1.0 if r["any_gold_cited"] else 0.0 for r in answerable_rows])
                if answerable_rows
                else None,
                "citation_validity_rate": mean([1.0 if r["citation_valid"] else 0.0 for r in found_rows])
                if found_rows
                else None,
                "not_found_rate_unanswerable": mean([1.0 if not r["found"] else 0.0 for r in unanswerable_rows])
                if unanswerable_rows
                else None,
            }
            result["latency"]["answer"] = _latency_stats(
                [r["answer_latency"] for r in answer_rows if r["answer_latency"] is not None]
            )

    return result, retrieval_rows


# --- Reporting -------------------------------------------------------------


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def print_report(results: list[dict], skips: list[str]) -> None:
    for msg in skips:
        print(f"  (skipped) {msg}")
    if skips:
        print()

    header = f"{'config':<22} {'n':>4} {'R@1':>6} {'R@5':>6} {'R@20':>6} {'MRR@10':>7} {'retr p50/p95':>16} {'rerank p50/p95':>16}"
    print(header)
    print("-" * len(header))
    for r in results:
        o = r["retrieval"]["overall"]
        lat = r["latency"]
        retr = lat["retrieve"]
        rer = lat["rerank"]
        retr_s = f"{_fmt(retr['p50'])}/{_fmt(retr['p95'])}" if retr else "n/a"
        rer_s = f"{_fmt(rer['p50'])}/{_fmt(rer['p95'])}" if rer else "n/a"
        print(
            f"{r['config']:<22} {o['n']:>4} {_fmt(o['recall_at_1']):>6} {_fmt(o['recall_at_5']):>6} "
            f"{_fmt(o['recall_at_20']):>6} {_fmt(o['mrr_at_10']):>7} {retr_s:>16} {rer_s:>16}"
        )
    print()

    for r in results:
        if not r["retrieval"]["by_type"]:
            continue
        print(f"  {r['config']} by type:")
        for t, m in r["retrieval"]["by_type"].items():
            print(f"    {t:<18} n={m['n']:<3} R@1={_fmt(m['recall_at_1'])} R@5={_fmt(m['recall_at_5'])} R@20={_fmt(m['recall_at_20'])} MRR@10={_fmt(m['mrr_at_10'])}")
    print()

    has_answers = any(r.get("answers") for r in results)
    if has_answers:
        print(f"{'config':<22} {'cite acc':>9} {'cite valid':>11} {'not-found rate':>15} {'answer p50/p95':>16}")
        for r in results:
            a = r.get("answers")
            if not a:
                reason = r.get("answers_skip_reason", "no answer check")
                print(f"{r['config']:<22} (skipped: {reason})")
                continue
            ans_lat = r["latency"].get("answer")
            ans_s = f"{_fmt(ans_lat['p50'])}/{_fmt(ans_lat['p95'])}" if ans_lat else "n/a"
            print(
                f"{r['config']:<22} {_fmt(a['citation_accuracy']):>9} {_fmt(a['citation_validity_rate']):>11} "
                f"{_fmt(a['not_found_rate_unanswerable']):>15} {ans_s:>16}"
            )
        print()


def write_failures_csv(path: Path, results: list[dict], golden_by_id: dict[str, GoldenItem], rows_by_config: dict[str, dict[str, dict]]) -> int:
    n_written = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "id", "type", "competition", "question", "gold", "rank", f"top_{FAILURE_TOP_N}_retrieved"])
        for r in results:
            cfg_name = r["config"]
            rows = rows_by_config[cfg_name]
            for item_id, row in rows.items():
                item = golden_by_id[item_id]
                if item.type == "unanswerable":
                    continue
                rank = row["rank"]
                if rank is not None and rank <= FAILURE_TOP_N:
                    continue
                writer.writerow(
                    [
                        cfg_name,
                        item.id,
                        item.type,
                        item.competition,
                        item.question,
                        ";".join(f"{g.doc_id}:{g.clause}" for g in item.gold),
                        rank if rank is not None else "not_found",
                        ";".join(f"{d}:{c}" for d, c in row["top"]),
                    ]
                )
                n_written += 1
    return n_written


def _index_version() -> str:
    path = BASE_DIR / CHROMA_DIR / "index_version.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH, help="path to a golden JSONL file")
    parser.add_argument("--configs", type=str, default=None, help="comma-separated config names (default: all available)")
    parser.add_argument("--with-answers", action="store_true", help="also run the full pipeline and check answers")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N golden rows")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="where to write results JSON/CSV")
    args = parser.parse_args()

    configure_logging()

    if not args.golden.exists():
        logger.error(
            "%s not found. Copy eval/golden_template.jsonl to eval/golden.jsonl and fill it in by hand "
            "(PROJECT_PLAN.md Phase 3) -- this file is never generated automatically.",
            args.golden,
        )
        return 1

    golden = load_golden(args.golden)
    if args.limit:
        golden = golden[: args.limit]
    if not golden:
        logger.error("%s has no rows", args.golden)
        return 1

    all_configs, skips = available_configs()
    if args.configs:
        requested = [name.strip() for name in args.configs.split(",") if name.strip()]
        unknown = [name for name in requested if name not in all_configs]
        if unknown:
            logger.error("Unknown or unavailable config(s): %s. Available: %s", unknown, sorted(all_configs))
            return 1
        configs = [all_configs[name] for name in requested]
    else:
        configs = list(all_configs.values())

    logger.info("Evaluating %d configs on %d golden questions from %s", len(configs), len(golden), args.golden)

    golden_by_id = {item.id: item for item in golden}
    results: list[dict] = []
    retrieval_rows_by_config: dict[str, dict[str, dict]] = {}

    for cfg in configs:
        logger.info("Running config %r ...", cfg.name)
        result, rows = evaluate_config(cfg, golden, args.with_answers)
        retrieval_rows_by_config[cfg.name] = rows
        results.append(result)

    print_report(results, skips)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    payload = {
        "timestamp": timestamp,
        "golden_path": str(args.golden),
        "n_questions": len(golden),
        "index_version": _index_version(),
        "prompt_version": PROMPT_VERSION,
        "with_answers": args.with_answers,
        "skipped_configs": skips,
        "results": results,
    }
    results_path = args.out_dir / f"{timestamp}.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    failures_path = args.out_dir / f"{timestamp}_failures.csv"
    n_failures = write_failures_csv(failures_path, results, golden_by_id, retrieval_rows_by_config)

    print(f"Results written to {results_path}")
    print(f"Failures ({n_failures} rows, recall@{FAILURE_TOP_N} misses) written to {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
