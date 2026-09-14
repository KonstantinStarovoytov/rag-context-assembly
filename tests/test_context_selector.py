from langchain_core.documents import Document

from src.rag.context_selector import select_generation_context
from src.rag.reranker import RerankResult


def _result(source: str, heading: str, rank: int) -> RerankResult:
    return RerankResult(
        document=Document(
            page_content=str(rank), metadata={"source": source, "h2": heading}
        ),
        retrieval_score=1.0,
        rerank_score=1.0,
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
