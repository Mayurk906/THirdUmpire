from src.query_analyzer import build_query_plan, detect_compared_competitions, detect_competition


def test_compared_competitions_includes_bare_test_alongside_other_formats() -> None:
    question = "How does the minimum over rate differ between Test, ODI and T20I cricket?"
    assert detect_compared_competitions(question) == ["icc_odi", "icc_t20i", "icc_test"]
    plan = build_query_plan(question)
    assert plan.allowed_competitions is None
    assert plan.compared_competitions == ["icc_odi", "icc_t20i", "icc_test"]


def test_compared_competitions_none_for_single_format_or_domestic() -> None:
    assert detect_compared_competitions("What is the ODI powerplay?") is None
    assert detect_compared_competitions("Is a bare test of the bat allowed?") is None
    assert detect_compared_competitions("Ranji Trophy vs ODI follow-on") is None


def test_override_disables_comparison() -> None:
    plan = build_query_plan("Test vs ODI over rate", competition_override="icc_odi")
    assert plan.compared_competitions is None


def test_detects_ipl_by_abbreviation() -> None:
    assert detect_competition("Is a runner allowed in IPL?") == "ipl"


def test_detects_ipl_by_full_name_case_insensitive() -> None:
    assert detect_competition("what happens in the indian premier league") == "ipl"


def test_detects_t20i_by_abbreviation() -> None:
    assert detect_competition("What's the over limit in a T20I?") == "icc_t20i"


def test_detects_t20i_by_full_name() -> None:
    assert detect_competition("Rules for a T20 International match") == "icc_t20i"


def test_no_competition_detected_returns_none() -> None:
    assert detect_competition("Is the batter out if bowled?") is None


def test_detects_icc_test_match() -> None:
    assert detect_competition("What's the follow-on rule in a Test match?") == "icc_test"


def test_detects_icc_odi_by_abbreviation() -> None:
    assert detect_competition("How many overs per bowler in an ODI?") == "icc_odi"


def test_detects_icc_odi_by_full_name() -> None:
    assert detect_competition("Powerplay rules in a One-day International") == "icc_odi"


def test_detects_bcci_domestic_multiday_by_ranji_trophy() -> None:
    assert detect_competition("Follow-on rules in the Ranji Trophy") == "bcci_domestic_multiday"


def test_detects_bcci_domestic_odi_by_vijay_hazare() -> None:
    assert detect_competition("Powerplay overs in the Vijay Hazare Trophy") == "bcci_domestic_odi"


def test_detects_bcci_domestic_t20_by_syed_mushtaq_ali() -> None:
    assert detect_competition("Super over rules in Syed Mushtaq Ali Trophy") == "bcci_domestic_t20"


def test_domestic_tournament_name_takes_priority_over_generic_test_pattern() -> None:
    # "Ranji Trophy" questions won't usually say "Test match" too, but the
    # domestic-specific pattern must still win if they do.
    assert detect_competition("Ranji Trophy Test match follow-on rules") == "bcci_domestic_multiday"


def test_question_naming_multiple_generic_formats_is_ambiguous() -> None:
    # Regression test for a real bug caught via live usage: this question
    # was matching "T20I" first (whichever generic pattern happened to be
    # listed first) and silently restricting retrieval to T20I-only,
    # excluding the Test and ODI documents -- producing a confident "not
    # found" for a real, answerable comparison question instead of actually
    # comparing all three. Naming several formats must fall back to no
    # filter (search everything), not an arbitrary pick.
    assert detect_competition("How does the minimum over rate differ between Test, ODI and T20I cricket?") is None
    assert detect_competition("Is IPL's over rate the same as a T20I's?") is None


def test_question_naming_one_generic_format_among_several_mentions_still_detects_it() -> None:
    # Only one of the generic patterns actually matches here (ODI) even
    # though the question is long -- must not be treated as ambiguous.
    assert detect_competition("What is the powerplay length in an ODI match?") == "icc_odi"


def test_query_plan_ipl_filters_ipl_and_laws() -> None:
    plan = build_query_plan("In IPL, can a runner bat?")
    assert plan.competition == "ipl"
    assert plan.allowed_competitions == ["ipl", "laws"]


def test_query_plan_t20i_filters_icc_t20i_and_laws() -> None:
    plan = build_query_plan("In a T20I, can a runner bat?")
    assert plan.competition == "icc_t20i"
    assert plan.allowed_competitions == ["icc_t20i", "laws"]


def test_query_plan_no_competition_has_no_filter() -> None:
    plan = build_query_plan("Can a runner bat?")
    assert plan.competition is None
    assert plan.allowed_competitions is None


def test_query_plan_override_wins_over_detected_competition() -> None:
    plan = build_query_plan("In IPL, can a runner bat?", competition_override="laws")
    assert plan.competition == "laws"
    assert plan.allowed_competitions == ["laws"]


def test_query_plan_bcci_domestic_multiday_filters_competition_and_laws() -> None:
    plan = build_query_plan("Follow-on rules in the Ranji Trophy")
    assert plan.competition == "bcci_domestic_multiday"
    assert plan.allowed_competitions == ["bcci_domestic_multiday", "laws"]
