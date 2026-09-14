from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr
    openai_embedding_model: str = "text-embedding-3-small"

    cohere_api_key: SecretStr
    cohere_rerank_model: str = "rerank-v4.0-fast"

    qdrant_url: str = "http://localhost:6333"
    # Qdrant Cloud requires a key; a local container does not.
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "agent_docs_v2"
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    answer_prompt_version: int | None = Field(default=None, ge=1)
    answer_evaluator_prompt_version: int | None = Field(default=None, ge=1)
    translate_prompt_version: int | None = Field(default=None, ge=1)
    evidence_planner_prompt_version: int | None = Field(default=None, ge=1)
    prompt_strict: bool = False
    openai_chat_model: str = "gpt-5.6-luna"
    tracing_enabled: bool = True
    # 5 -> 8 covered both aspects in half the multi-aspect set instead of a third.
    generation_top_k: int = Field(default=8, ge=1)
    retrieval_strategy: Literal["dense", "hybrid", "hybrid-english"] = "hybrid"
    translate_non_english: bool = True
    retrieval_top_k: int = Field(default=10, ge=1)
    per_query_top_k: int = Field(default=20, ge=1)
    qdrant_hybrid_collection: str = "agent_docs_hybrid_v1"
    sparse_embedding_model: str = "Qdrant/bm25"

    # Every served request spends OpenAI and Cohere credits, so the deployed API
    # refuses to start without a token. Unset is allowed only for local CLI use.
    api_token: SecretStr | None = None
    api_max_question_chars: int = Field(default=500, ge=1)


settings = Settings()
