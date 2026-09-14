from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.models import SourceDocument

HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def chunk_document(
    document: SourceDocument,
) -> list[Document]:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=False,
    )

    sections = markdown_splitter.split_text(document.content)

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

    return text_splitter.split_documents(sections)


def chunk_documents(
    documents: list[SourceDocument],
) -> list[Document]:
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    return chunks
