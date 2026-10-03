from __future__ import annotations

from src import query_planner
from src.query_planner import _CompetitionList, build_query_plan_llm


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self._result)


def test_single_competition(monkeypatch):
    monkeypatch.setattr(
        query_planner, "_get_llm", lambda: _FakeLLM(_CompetitionList(competitions=["icc_odi"], gender="unspecified"))
    )
    plan = build_query_plan_llm("What's the powerplay in an ODI?")
    assert plan.competition == "icc_odi"
    assert plan.allowed_competitions == ["icc_odi", "laws"]
    assert plan.compared_competitions is None
    assert plan.gender is None


def test_comparison_sets_compared_competitions_and_no_single_filter(monkeypatch):
    monkeypatch.setattr(
        query_planner,
        "_get_llm",
        lambda: _FakeLLM(_CompetitionList(competitions=["icc_test", "icc_odi", "icc_t20i"], gender="unspecified")),
    )
    plan = build_query_plan_llm("How does the over rate differ between Test, ODI and T20I?")
    assert plan.competition is None
    assert plan.allowed_competitions is None
    assert plan.compared_competitions == ["icc_odi", "icc_t20i", "icc_test"]


def test_gender_is_passed_through(monkeypatch):
    monkeypatch.setattr(
        query_planner, "_get_llm", lambda: _FakeLLM(_CompetitionList(competitions=["icc_t20i"], gender="women"))
    )
    plan = build_query_plan_llm("What's the powerplay in women's T20Is?")
    assert plan.gender == "women"


def test_no_competition_named_returns_unfiltered_plan(monkeypatch):
    monkeypatch.setattr(query_planner, "_get_llm", lambda: _FakeLLM(_CompetitionList(competitions=[])))
    plan = build_query_plan_llm("What is a no-ball?")
    assert plan.competition is None
    assert plan.allowed_competitions is None
    assert plan.compared_competitions is None


def test_llm_failure_falls_back_to_regex_detector(monkeypatch):
    monkeypatch.setattr(query_planner, "_get_llm", lambda: _FakeLLM(RuntimeError("429 quota exceeded")))
    plan = build_query_plan_llm("Is a runner allowed in IPL?")
    assert plan.competition == "ipl"  # same result query_analyzer.detect_competition() gives


def test_competition_override_skips_the_llm_entirely(monkeypatch):
    def _boom():
        raise AssertionError("the LLM must not be called when an override is given")

    monkeypatch.setattr(query_planner, "_get_llm", _boom)
    plan = build_query_plan_llm("Any question", competition_override="laws")
    assert plan.competition == "laws"
