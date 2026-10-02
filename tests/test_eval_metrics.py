from eval.metrics import hit_rank, mean, percentile, reciprocal_rank, recall_at_k


def test_hit_rank_finds_first_match() -> None:
    ranked = [("a", "1"), ("b", "2"), ("c", "3")]
    assert hit_rank(ranked, {("c", "3")}) == 3
    assert hit_rank(ranked, {("b", "2"), ("c", "3")}) == 2


def test_hit_rank_none_when_no_match() -> None:
    ranked = [("a", "1"), ("b", "2")]
    assert hit_rank(ranked, {("z", "9")}) is None


def test_hit_rank_empty_ranked_list() -> None:
    assert hit_rank([], {("a", "1")}) is None


def test_recall_at_k() -> None:
    assert recall_at_k(1, 1) is True
    assert recall_at_k(5, 5) is True
    assert recall_at_k(6, 5) is False
    assert recall_at_k(None, 5) is False


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(10) == 0.1
    assert reciprocal_rank(11) == 0.0  # outside the default k=10 window
    assert reciprocal_rank(None) == 0.0


def test_reciprocal_rank_custom_k() -> None:
    assert reciprocal_rank(3, k=3) == 1.0 / 3
    assert reciprocal_rank(4, k=3) == 0.0


def test_mean() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0


def test_percentile_matches_known_values() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 0) == 10.0
    assert percentile(values, 50) == 30.0
    assert percentile(values, 100) == 50.0


def test_percentile_interpolates() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    # rank = (75/100) * 3 = 2.25 -> between index 2 (3.0) and 3 (4.0)
    assert percentile(values, 75) == 3.25


def test_percentile_single_value() -> None:
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0


def test_percentile_unsorted_input() -> None:
    assert percentile([5.0, 1.0, 3.0], 50) == 3.0


def test_percentile_empty_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        percentile([], 50)
