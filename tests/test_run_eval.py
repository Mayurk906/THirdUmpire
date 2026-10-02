from pathlib import Path

import pytest

from eval import run_eval
from eval.run_eval import EvalConfig, GoldenItem, available_configs, evaluate_config, load_golden


def make_item(id_, type_="lookup", competition="laws", gold=None):
    return GoldenItem(
        id=id_,
        question=f"question {id_}",
        competition=competition,
        gold=gold if gold is not None else [{"doc_id": "doc", "clause": "1.1"}],
        answer_short="short",
        type=type_,
    )


def test_load_golden_parses_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id": "g1", "question": "Q1?", "competition": "laws", "gold": [{"doc_id": "d", "clause": "1.1"}], '
        '"answer_short": "a", "type": "lookup", "notes": ""}\n'
        '{"id": "g2", "question": "Q2?", "competition": "any", "gold": [], "answer_short": "", '
        '"type": "unanswerable", "notes": ""}\n',
        encoding="utf-8",
    )
    items = load_golden(path)
    assert [i.id for i in items] == ["g1", "g2"]
    assert items[0].gold_pairs() == {("d", "1.1")}
    assert items[1].competition_override() is None


def test_load_golden_raises_with_context_on_bad_row(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "g1"}\n', encoding="utf-8")  # missing required fields
    with pytest.raises(ValueError, match=r"golden\.jsonl:1"):
        load_golden(path)


def test_load_golden_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '\n{"id": "g1", "question": "Q?", "competition": "laws", "gold": [], "answer_short": "", '
        '"type": "lookup", "notes": ""}\n\n',
        encoding="utf-8",
    )
    items = load_golden(path)
    assert len(items) == 1


def test_competition_override_passes_through_specific_competition() -> None:
    item = make_item("g1", competition="ipl")
    assert item.competition_override() == "ipl"


def test_available_configs_always_includes_dense_baselines() -> None:
    configs, _ = available_configs()
    assert "dense" in configs
    assert "dense+base_rerank" in configs
    assert configs["dense"].reranker is None
    assert configs["dense+base_rerank"].reranker == "base"


def test_available_configs_includes_hybrid() -> None:
    # src/bm25.py (Phase 5) is a committed code file, so this is
    # deterministic across any checkout -- unlike the fine-tuned reranker
    # below, which is a gitignored local artifact.
    configs, _ = available_configs()
    assert "hybrid" in configs
    assert "hybrid+base_rerank" in configs


def test_available_configs_includes_ft_rerank_when_model_present(monkeypatch: pytest.MonkeyPatch) -> None:
    # models/reranker-ft/ (Phase 6) is gitignored, so this is monkeypatched
    # rather than relying on it actually existing on disk -- that would
    # pass or fail depending on whether this machine happens to have a
    # locally-downloaded fine-tuned model, not on the code being correct.
    monkeypatch.setattr(run_eval, "_ft_reranker_available", lambda: True)
    configs, skips = available_configs()
    assert "dense+ft_rerank" in configs
    assert skips == []


def test_available_configs_skips_ft_rerank_when_model_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "_ft_reranker_available", lambda: False)
    configs, skips = available_configs()
    assert "dense+ft_rerank" not in configs
    assert any("reranker" in s for s in skips)


def test_evaluate_config_computes_recall_and_mrr(monkeypatch: pytest.MonkeyPatch) -> None:
    golden = [
        make_item("g1", type_="lookup"),
        make_item("g2", type_="lookup"),
        make_item("g3", type_="unanswerable", gold=[]),
    ]
    fake_rows = {
        "g1": {"rank": 1, "retrieve_latency": 0.1, "rerank_latency": None, "top": []},
        "g2": {"rank": None, "retrieve_latency": 0.2, "rerank_latency": None, "top": []},
        "g3": {"rank": None, "retrieve_latency": 0.1, "rerank_latency": None, "top": []},
    }
    monkeypatch.setattr(run_eval, "eval_retrieval", lambda cfg, item: fake_rows[item.id])

    cfg = EvalConfig("dense", "dense", None)
    result, rows = evaluate_config(cfg, golden, with_answers=False)

    # g3 is unanswerable and excluded from retrieval accuracy; of the 2
    # answerable questions only g1 hit -> recall@1 = recall@20 = mrr@10 = 0.5.
    overall = result["retrieval"]["overall"]
    assert overall["n"] == 2
    assert overall["recall_at_1"] == 0.5
    assert overall["recall_at_20"] == 0.5
    assert overall["mrr_at_10"] == 0.5
    assert rows["g3"]["rank"] is None


def test_evaluate_config_skips_answers_without_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    golden = [make_item("g1")]
    monkeypatch.setattr(
        run_eval,
        "eval_retrieval",
        lambda cfg, item: {"rank": 1, "retrieve_latency": 0.1, "rerank_latency": None, "top": []},
    )
    cfg = EvalConfig("dense", "dense", None)  # no reranker -> run_query() can't be used
    result, _ = evaluate_config(cfg, golden, with_answers=True)
    assert result["answers"] is None
    assert "answers_skip_reason" in result


def test_evaluate_config_with_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    golden = [make_item("g1", type_="lookup"), make_item("g2", type_="unanswerable", gold=[])]
    monkeypatch.setattr(
        run_eval,
        "eval_retrieval",
        lambda cfg, item: {"rank": 1, "retrieve_latency": 0.1, "rerank_latency": 0.05, "top": []},
    )

    def fake_eval_answer(cfg, item):
        unanswerable = item.type == "unanswerable"
        return {
            "found": not unanswerable,
            "citation_valid": True,
            "any_gold_cited": not unanswerable,
            "retrieve_latency": 0.1,
            "rerank_latency": 0.05,
            "answer_latency": 0.3,
        }

    monkeypatch.setattr(run_eval, "eval_answer", fake_eval_answer)

    cfg = EvalConfig("dense+base_rerank", "dense", "base")
    result, _ = evaluate_config(cfg, golden, with_answers=True)

    assert result["answers"]["citation_accuracy"] == 1.0
    assert result["answers"]["citation_validity_rate"] == 1.0
    assert result["answers"]["not_found_rate_unanswerable"] == 1.0
