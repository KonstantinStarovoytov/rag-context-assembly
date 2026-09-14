from langchain_core.documents import Document
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient

from src.config import settings
from src.rag.embeddings import (
    get_embeddings,
    get_sparse_embeddings,
)


def create_vector_store() -> QdrantVectorStore:
    client = QdrantClient(url=settings.qdrant_url)

    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )


def get_hybrid_vector_store() -> QdrantVectorStore:
    client = QdrantClient(url=settings.qdrant_url)

    return QdrantVectorStore(
        client=client,
        collection_name=(settings.qdrant_hybrid_collection),
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
    )


def add_documents(
    documents: list[Document],
) -> list[str]:
    store = create_vector_store()

    return store.add_documents(documents)


def get_vector_store() -> QdrantVectorStore:
    client = QdrantClient(
        url=settings.qdrant_url,
    )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )
