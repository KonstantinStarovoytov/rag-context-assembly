"""Every paid client is bounded; a hung provider must not hold a thread for 30 min."""

from typing import Any

import pytest

from src.rag import embeddings, llm, reranker


def test_chat_model_is_bounded() -> None:
    model = llm.chat_model()

    assert model.request_timeout == llm.CHAT_TIMEOUT_SECONDS
    assert model.max_retries == 1


def test_embeddings_are_bounded() -> None:
    model = embeddings.get_embeddings()

    assert model.request_timeout == llm.EMBED_TIMEOUT_SECONDS
    assert model.max_retries == 1


def test_cohere_client_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_client(**kwargs: Any) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(reranker.cohere, "ClientV2", fake_client)

    reranker.CohereReranker()

    assert seen["timeout"] == reranker.RERANK_TIMEOUT_SECONDS
    assert seen["max_retries"] == 1
