"""Select a useful, non-repetitive subset of reranked evidence for generation."""

from src.rag.reranker import RerankResult

MAX_CHUNKS_PER_SECTION = 2


def _section_key(result: RerankResult) -> str:
    metadata = result.document.metadata
    heading = " > ".join(
        value
        for value in (metadata.get("h1"), metadata.get("h2"), metadata.get("h3"))
        if value
    )
    return f"{metadata.get('source') or ''}\u0000{heading}"


def select_generation_context(
    results: list[RerankResult],
    top_k: int,
    *,
    max_per_section: int = MAX_CHUNKS_PER_SECTION,
    min_rerank_score: float | None = None,
) -> list[RerankResult]:
    """Preserve rank while preventing one repeated section consuming all context.

    A second pass backfills the requested context size when the candidate set has
    too few distinct sections. This keeps a focused single-section answer intact.

    `min_rerank_score` drops weak candidates before either pass runs, so a
    document with many distinct but weakly-scored sections cannot fill the
    context just by dodging the section cap (observed: a single unrelated
    document supplying 5 of 5 slots via 5 different headings, scores
    0.28-0.53, well below the ~0.35 floor measured on genuinely relevant
    multi-aspect context). Returning fewer than top_k chunks is preferable to
    padding with a weak match. None (the default) preserves prior behaviour.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if max_per_section < 1:
        raise ValueError("max_per_section must be positive")

    if min_rerank_score is not None:
        results = [r for r in results if r.rerank_score >= min_rerank_score]

    selected: list[RerankResult] = []
    section_counts: dict[str, int] = {}
    for result in results:
        section = _section_key(result)
        if section_counts.get(section, 0) >= max_per_section:
            continue
        selected.append(result)
        section_counts[section] = section_counts.get(section, 0) + 1
        if len(selected) == top_k:
            return selected

    # A narrow topic can legitimately have evidence from one document. Fill the
    # remaining slots in original rank order instead of returning too little context.
    for result in results:
        if result in selected:
            continue
        selected.append(result)
        if len(selected) == top_k:
            break
    return selected
