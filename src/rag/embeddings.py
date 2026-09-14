from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse

from src.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )


def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(
        model_name=settings.sparse_embedding_model,
    )
