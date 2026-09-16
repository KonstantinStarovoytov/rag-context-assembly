"""Chunking: cleaning, section splitting, and the embedded contextual prefix."""

from src.ingestion.chunker import chunk_document
from src.models import SourceDocument


def _doc(content: str, **overrides: str) -> SourceDocument:
    base = dict(
        title="Hooks reference",
        content=content,
        url="https://code.claude.com/docs/en/hooks.md",
        vendor="anthropic",
        product="claude-code",
    )
    base.update(overrides)
    return SourceDocument(**base)  # type: ignore[arg-type]


def test_boilerplate_is_cleaned_before_splitting() -> None:
    doc = _doc(
        "> ## Documentation Index\n"
        "> Fetch the complete documentation index at: https://code.claude.com/docs/llms.txt\n"
        "> Use this file to discover all available pages before exploring further.\n"
        "\n"
        "# Hooks reference\n\nReal content.\n"
    )

    chunks = chunk_document(doc)

    assert all("Documentation Index" not in c.metadata["raw_content"] for c in chunks)
    assert all("llms.txt" not in c.metadata["raw_content"] for c in chunks)


def test_page_content_is_prefixed_for_embedding_and_bm25() -> None:
    doc = _doc(
        "# Hooks reference\n\n## Configuration\n\nSet ttlMs to control caching.\n"
    )

    chunks = chunk_document(doc)

    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk.page_content.startswith(
        "Vendor: anthropic\nProduct: claude-code\nDocument: Hooks reference\n"
    )
    assert "Set ttlMs to control caching." in chunk.page_content


def test_raw_content_metadata_has_no_prefix() -> None:
    doc = _doc("# Hooks reference\n\nSet ttlMs to control caching.\n")

    chunk = chunk_document(doc)[0]

    assert chunk.metadata["raw_content"] == chunk.page_content.split("\n\n", 1)[-1]
    assert not chunk.metadata["raw_content"].startswith("Vendor:")


def test_raw_content_is_what_generator_and_reranker_should_display() -> None:
    """The contextual prefix must not duplicate the labels generator.py and
    reranker.py already add themselves from the same metadata."""
    doc = _doc("# Hooks reference\n\n## Configuration\n\nSet ttlMs.\n")

    chunk = chunk_document(doc)[0]

    assert "Vendor:" not in chunk.metadata["raw_content"]
    assert "Product:" not in chunk.metadata["raw_content"]
