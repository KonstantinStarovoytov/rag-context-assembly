import math
from functools import partial

from langfuse import Evaluation


def _is_relevant(
    result,
    expected_output,
):
    for target in expected_output["relevant"]:
        vendor = target.get("vendor")

        if vendor and result.get("vendor") != vendor:
            continue

        product = target.get("product")

        if product and result.get("product") != product:
            continue

        title = target.get("title")

        if title and result.get("title") != title:
            continue

        heading_contains = target.get("heading_contains")

        if (
            heading_contains
            and heading_contains.lower()
            not in result.get(
                "heading",
                "",
            ).lower()
        ):
            continue

        source_contains = target.get("source_contains")

        if (
            source_contains
            and source_contains.lower()
            not in result.get(
                "source",
                "",
            ).lower()
        ):
            continue

        return True

    return False


def hit_at_1(
    *,
    output,
    expected_output,
    **kwargs,
):
    results = output["results"]

    hit = bool(
        results
        and _is_relevant(
            results[0],
            expected_output,
        )
    )

    return Evaluation(
        name="hit_at_1",
        value=float(hit),
    )


def hit_at_5(
    *,
    output,
    expected_output,
    **kwargs,
):
    results = output["results"][:5]

    hit = any(
        _is_relevant(
            result,
            expected_output,
        )
        for result in results
    )

    return Evaluation(
        name="hit_at_5",
        value=float(hit),
    )


def hit_at_10(
    *,
    output,
    expected_output,
    **kwargs,
):
    hit = any(
        _is_relevant(
            result,
            expected_output,
        )
        for result in output["results"][:10]
    )

    return Evaluation(
        name="hit_at_10",
        value=float(hit),
    )


def reciprocal_rank(
    *,
    output,
    expected_output,
    **kwargs,
):
    for rank, result in enumerate(
        output["results"],
        start=1,
    ):
        if _is_relevant(
            result,
            expected_output,
        ):
            return Evaluation(
                name="reciprocal_rank",
                value=1.0 / rank,
                comment=f"Relevant result at rank {rank}",
            )

    return Evaluation(
        name="reciprocal_rank",
        value=0.0,
        comment="Relevant result not found",
    )


# hit@k saturates once one relevant chunk sits anywhere in the top k, which is
# the case on every current dataset. These two keep moving: precision counts
# how much of the window is relevant, NDCG rewards putting it first.


def precision_at_k(*, output, expected_output, k, **kwargs):
    """Relevant results among the first k, over the results actually present."""
    results = output["results"][:k]
    if not results:
        return Evaluation(name=f"precision_at_{k}", value=0.0, comment="No results")
    relevant = sum(_is_relevant(result, expected_output) for result in results)
    return Evaluation(
        name=f"precision_at_{k}",
        value=relevant / len(results),
        comment=f"{relevant} of {len(results)} relevant",
    )


def ndcg_at_k(*, output, expected_output, k, **kwargs):
    """Binary-gain NDCG. The ideal ranking has every relevant result first;
    the number of relevant results is taken from the full list, since the
    datasets do not enumerate all relevant chunks."""
    results = output["results"]
    gains = [float(_is_relevant(result, expected_output)) for result in results]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains[:k], 1))
    ideal_hits = min(k, int(sum(gains)))
    if ideal_hits == 0:
        return Evaluation(name=f"ndcg_at_{k}", value=0.0, comment="No relevant result")
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return Evaluation(name=f"ndcg_at_{k}", value=dcg / idcg)


def _named(fn, k):
    # Langfuse labels evaluator errors by __name__, which partial lacks.
    bound = partial(fn, k=k)
    bound.__name__ = f"{fn.__name__.removesuffix('_k')}_{k}"
    return bound


precision_at_5 = _named(precision_at_k, 5)
precision_at_8 = _named(precision_at_k, 8)
precision_at_10 = _named(precision_at_k, 10)
ndcg_at_10 = _named(ndcg_at_k, 10)
