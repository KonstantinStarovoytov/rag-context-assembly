"""Text sent to Cohere for reranking."""

from langchain_core.documents import Document

from src.rag.reranker import CohereReranker


def test_document_text_uses_raw_content_not_the_embedding_prefix() -> None:
    """page_content may carry the embedded contextual prefix; this method
    builds its own Title/Section/Product header and must not duplicate it."""
    document = Document(
        page_content="Vendor: a\nProduct: b\nDocument: c\nSection: d\n\nreal text",
        metadata={"title": "c", "product": "b", "raw_content": "real text"},
    )

    text = CohereReranker._document_text(document)

    assert text.count("Product: b") == 1
    assert "Vendor: a\nProduct: b\nDocument: c" not in text
    assert "real text" in text


def test_document_text_falls_back_to_page_content_without_raw_content() -> None:
    document = Document(page_content="plain text", metadata={"title": "t"})

    text = CohereReranker._document_text(document)

    assert "plain text" in text
