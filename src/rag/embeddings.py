from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse

from src.config import settings
from src.rag.llm import EMBED_TIMEOUT_SECONDS


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        request_timeout=EMBED_TIMEOUT_SECONDS,
        max_retries=1,
    )


def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(
        model_name=settings.sparse_embedding_model,
    )
