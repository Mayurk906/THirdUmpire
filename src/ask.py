"""CLI entry point.

Usage: python -m src.ask "question" [--competition ipl|icc_t20i|laws] [--reranker base|ft] [--debug]
"""
from __future__ import annotations

import argparse

from src.config import DEFAULT_FIRST_STAGE, configure_logging
from src.pipeline import run_query
from src.schemas import KNOWN_COMPETITIONS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--competition", choices=list(KNOWN_COMPETITIONS), default=None)
    parser.add_argument("--first-stage", choices=["dense", "hybrid"], default=DEFAULT_FIRST_STAGE)
    parser.add_argument("--reranker", choices=["base", "ft"], default="base")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging()
    response = run_query(
        args.question,
        competition=args.competition,
        first_stage=args.first_stage,
        reranker=args.reranker,
        debug=args.debug,
    )

    print(f"\nQuestion: {args.question}")
    print(f"Detected/selected competition: {response.query_plan.competition or 'none'}")
    print(f"\nAnswer: {response.answer.answer}")
    print(f"Found: {response.answer.found} | Applies to: {response.answer.applies_to}")

    if response.answer.citations:
        print("Citations:")
        for c in response.answer.citations:
            print(f"  - [{c.doc_id}, {c.clause}, p.{c.page}]")
    else:
        print("Citations: none")

    if not response.citation_valid:
        print("WARNING: one or more citations could not be validated against the retrieved context.")
        for c in response.invalid_citations:
            print(f"  INVALID: [{c.doc_id}, {c.clause}, p.{c.page}]")

    if args.debug:
        print("\n--- Retrieved (first stage) ---")
        for sc in response.retrieved or []:
            print(f"  {sc.score:.3f}  {sc.chunk.doc_id}  {sc.chunk.clause}  {sc.chunk.clause_title}")

        print("\n--- Reranked ---")
        for sc in response.reranked or []:
            raw = f"{sc.raw_score:.3f}" if sc.raw_score is not None else "n/a"
            print(f"  {sc.score:.3f} (raw {raw})  {sc.chunk.doc_id}  {sc.chunk.clause}  {sc.chunk.clause_title}")

        print("\n--- Latencies (s) ---")
        for stage, seconds in response.latencies.items():
            print(f"  {stage}: {seconds:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
