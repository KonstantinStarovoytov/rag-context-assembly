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
