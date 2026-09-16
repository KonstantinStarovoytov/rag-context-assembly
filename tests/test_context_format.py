"""Each chunk shown to the generator names the product it documents.

Regression for a real misattribution: without an explicit Product label the
generator inferred another product's conventions applied to the one asked
about (a Copilot question answered by citing a Claude Code memory.md section
as if it were about Copilot). Product is already in the chunk metadata from
ingestion; this only renders it.
"""

from langchain_core.documents import Document

from src.rag.generator import _format_context
from src.rag.reranker import RerankResult


def _result(vendor: str, product: str, title: str) -> RerankResult:
    return RerankResult(
        document=Document(
            page_content="content",
            metadata={
                "source": "https://docs.test/x.md",
                "title": title,
                "vendor": vendor,
                "product": product,
            },
        ),
        retrieval_score=1.0,
        rerank_score=1.0,
        original_rank=1,
        rerank_rank=1,
    )


def test_context_names_the_product_per_chunk() -> None:
    context = _format_context([_result("anthropic", "claude-code", "Memory")])

    assert "Product: anthropic/claude-code" in context


def test_context_format_tolerates_missing_product() -> None:
    result = RerankResult(
        document=Document(page_content="c", metadata={"source": "https://x"}),
        retrieval_score=1.0,
        rerank_score=1.0,
        original_rank=1,
        rerank_rank=1,
    )

    context = _format_context([result])

    assert "Product: " not in context


def test_context_uses_raw_content_not_the_embedding_prefix() -> None:
    """page_content may carry the Vendor/Product/Document/Section prefix used
    for embedding/BM25; the generator must show raw text, or its own Product
    line (already present) would be immediately followed by a duplicate."""
    result = RerankResult(
        document=Document(
            page_content="Vendor: a\nProduct: b\nDocument: c\nSection: d\n\nreal text",
            metadata={
                "source": "https://x",
                "vendor": "a",
                "product": "b",
                "raw_content": "real text",
            },
        ),
        retrieval_score=1.0,
        rerank_score=1.0,
        original_rank=1,
        rerank_rank=1,
    )

    context = _format_context([result])

    assert context.count("Product: a/b") == 1
    assert "Vendor: a\nProduct: b\nDocument: c" not in context
    assert "real text" in context
