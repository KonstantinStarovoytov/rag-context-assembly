from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.ingestion.clean import clean_markdown
from src.models import SourceDocument

HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Bumped whenever chunking/cleaning logic changes, so src.reindex can tell
# a page needs re-embedding even when its fetched markdown is unchanged.
CHUNKER_VERSION = "2-clean-and-contextualize"


def _heading_path(metadata: dict[str, str]) -> str:
    return " > ".join(
        value
        for value in (metadata.get("h1"), metadata.get("h2"), metadata.get("h3"))
        if value
    )


def _contextualize(chunk: Document, *, vendor: str, product: str, title: str) -> None:
    """Prefix the embedded/BM25-indexed text with document context.

    Without this, a chunk's text alone rarely names the product it documents
    (only the first chunk of a section keeps its Markdown heading; later
    chunks of a long section carry none), so BM25 and dense retrieval cannot
    tell apart same-named concepts across vendors (`hooks`, `mcp`, `rules`).
    The raw text is kept in metadata for callers that build their own
    Title/Section/Product display (generator.py, reranker.py) and must not
    show the label twice.
    """
    heading = _heading_path(chunk.metadata)
    chunk.metadata["raw_content"] = chunk.page_content
    chunk.page_content = (
        f"Vendor: {vendor}\n"
        f"Product: {product}\n"
        f"Document: {title}\n"
        f"Section: {heading}\n\n"
        f"{chunk.page_content}"
    )


def chunk_document(
    document: SourceDocument,
) -> list[Document]:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=False,
    )

    sections = markdown_splitter.split_text(clean_markdown(document.content))

    for section in sections:
        section.metadata.update(
            {
                "vendor": document.vendor,
                "product": document.product,
                "title": document.title,
                "source": document.url,
            }
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300,
    )

    chunks = text_splitter.split_documents(sections)
    for chunk in chunks:
        _contextualize(
            chunk,
            vendor=document.vendor,
            product=document.product,
            title=document.title,
        )
    return chunks


def chunk_documents(
    documents: list[SourceDocument],
) -> list[Document]:
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    return chunks
