"""Precision@k and NDCG@k: the metrics that still move when hit@k is saturated."""

from evals.evaluators import ndcg_at_k, precision_at_k

EXPECTED = {"relevant": [{"title": "a"}, {"title": "b"}]}


def _output(*titles: str) -> dict[str, list[dict[str, str]]]:
    return {"results": [{"title": title} for title in titles]}


def test_precision_counts_relevant_among_top_k() -> None:
    output = _output("a", "junk", "b", "junk")

    assert precision_at_k(output=output, expected_output=EXPECTED, k=4).value == 0.5
    assert precision_at_k(output=output, expected_output=EXPECTED, k=2).value == 0.5
    assert precision_at_k(output=output, expected_output=EXPECTED, k=1).value == 1.0


def test_precision_divides_by_the_results_it_has() -> None:
    """A short result list must not be punished for missing slots."""
    output = _output("a")

    assert precision_at_k(output=output, expected_output=EXPECTED, k=8).value == 1.0


def test_precision_of_empty_results_is_zero() -> None:
    assert (
        precision_at_k(output={"results": []}, expected_output=EXPECTED, k=8).value
        == 0.0
    )


def test_ndcg_rewards_relevant_results_placed_earlier() -> None:
    early = ndcg_at_k(output=_output("a", "b", "junk"), expected_output=EXPECTED, k=3)
    late = ndcg_at_k(output=_output("junk", "a", "b"), expected_output=EXPECTED, k=3)

    assert early.value == 1.0
    assert late.value < early.value
    assert late.value > 0.0


def test_ndcg_is_zero_without_a_relevant_result() -> None:
    assert ndcg_at_k(output=_output("junk"), expected_output=EXPECTED, k=3).value == 0.0


def test_named_evaluators_expose_the_k_used() -> None:
    from evals.evaluators import ndcg_at_10, precision_at_8

    output = _output("a", "junk")

    assert (
        precision_at_8(output=output, expected_output=EXPECTED).name == "precision_at_8"
    )
    assert ndcg_at_10(output=output, expected_output=EXPECTED).name == "ndcg_at_10"
