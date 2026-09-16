from langchain_core.documents import Document

from src.rag.context_selector import select_generation_context
from src.rag.reranker import RerankResult


def _result(
    source: str, heading: str, rank: int, *, score: float = 1.0
) -> RerankResult:
    return RerankResult(
        document=Document(
            page_content=str(rank), metadata={"source": source, "h2": heading}
        ),
        retrieval_score=1.0,
        rerank_score=score,
        original_rank=rank,
        rerank_rank=rank,
    )


def test_context_selector_limits_repeated_sections_when_alternatives_exist():
    results = [
        _result("a", "one", 1),
        _result("a", "one", 2),
        _result("a", "one", 3),
        _result("a", "two", 4),
    ]

    selected = select_generation_context(results, 3, max_per_section=2)

    assert [result.rerank_rank for result in selected] == [1, 2, 4]


def test_context_selector_backfills_when_only_one_section_is_available():
    results = [_result("a", "one", 1), _result("a", "one", 2), _result("a", "one", 3)]

    selected = select_generation_context(results, 3, max_per_section=1)

    assert [result.rerank_rank for result in selected] == [1, 2, 3]


def test_min_rerank_score_drops_weak_candidates_before_either_pass():
    """A distractor document with many weak-scoring sections must not fill the
    context just because it has enough distinct headings to dodge the section
    cap (the observed Opus-pricing bug: 5 different hooks.md sections, scores
    0.53/0.38/0.32/0.30/0.28, none from the section-repeat pass)."""
    results = [
        _result("hooks.md", "a", 1, score=0.53),
        _result("hooks.md", "b", 2, score=0.38),
        _result("hooks.md", "c", 3, score=0.32),
        _result("hooks.md", "d", 4, score=0.30),
        _result("hooks.md", "e", 5, score=0.28),
    ]

    selected = select_generation_context(results, 5, min_rerank_score=0.35)

    # 0.53 and 0.38 clear the floor measured on genuinely relevant context;
    # the bottom three (0.32/0.30/0.28) do not.
    assert [result.rerank_rank for result in selected] == [1, 2]


def test_min_rerank_score_never_drops_a_strong_candidate():
    results = [_result("a", "one", 1, score=0.9), _result("a", "two", 2, score=0.4)]

    selected = select_generation_context(results, 2, min_rerank_score=0.35)

    assert [result.rerank_rank for result in selected] == [1, 2]


def test_min_rerank_score_defaults_to_no_filtering():
    """Existing callers that do not pass the threshold see no behaviour change."""
    results = [_result("a", "one", 1, score=0.01)]

    selected = select_generation_context(results, 1)

    assert [result.rerank_rank for result in selected] == [1]
