"""Fallback Cohere key: the trial rerank quota is small and shared across
every eval and the live service; a second key keeps requests working while
the primary key is over its limit."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from cohere.errors.too_many_requests_error import TooManyRequestsError
from pydantic import SecretStr

from src.rag import reranker as reranker_module
from src.rag.reranker import CohereReranker
from src.rag.retriever import SearchResult


def _settings(primary: str, fallback: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        cohere_api_key=SecretStr(primary),
        cohere_api_key_fallback=SecretStr(fallback) if fallback else None,
        cohere_rerank_model="rerank-v4.0-fast",
    )


def _rerank_response(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        results=[SimpleNamespace(index=i, relevance_score=1.0) for i in range(n)]
    )


def _candidates(n: int) -> list[SearchResult]:
    from langchain_core.documents import Document

    return [
        SearchResult(document=Document(page_content=f"c{i}", metadata={}), score=1.0)
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _reset_sticky_state():
    reranker_module.reset_fallback_state()
    yield
    reranker_module.reset_fallback_state()


def test_falls_back_on_primary_quota_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reranker_module, "settings", _settings("primary", "fallback"))
    primary_client = Mock()
    primary_client.rerank.side_effect = TooManyRequestsError(headers={}, body={})
    fallback_client = Mock()
    fallback_client.rerank.return_value = _rerank_response(1)
    seen_keys: list[str] = []

    def fake_client_v2(*, api_key: str, **_kwargs: object) -> Mock:
        seen_keys.append(api_key)
        return primary_client if api_key == "primary" else fallback_client

    monkeypatch.setattr(reranker_module.cohere, "ClientV2", fake_client_v2)

    result = CohereReranker().rerank("q", _candidates(1))

    assert len(result) == 1
    primary_client.rerank.assert_called_once()
    fallback_client.rerank.assert_called_once()


def test_sticks_to_fallback_after_primary_is_marked_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once one request has seen the primary key's quota error, later
    requests (new CohereReranker instances, as pipeline.py creates) go
    straight to the fallback instead of paying for a doomed primary call."""
    monkeypatch.setattr(reranker_module, "settings", _settings("primary", "fallback"))
    primary_client = Mock()
    primary_client.rerank.side_effect = TooManyRequestsError(headers={}, body={})
    fallback_client = Mock()
    fallback_client.rerank.return_value = _rerank_response(1)
    monkeypatch.setattr(
        reranker_module.cohere,
        "ClientV2",
        lambda *, api_key, **_k: (
            primary_client if api_key == "primary" else fallback_client
        ),
    )

    CohereReranker().rerank("q", _candidates(1))  # trips the sticky flag
    primary_client.rerank.reset_mock()

    CohereReranker().rerank("q", _candidates(1))  # a fresh instance, same process

    primary_client.rerank.assert_not_called()
    assert fallback_client.rerank.call_count == 2


def test_without_a_fallback_key_the_original_error_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reranker_module, "settings", _settings("primary", None))
    primary_client = Mock()
    primary_client.rerank.side_effect = TooManyRequestsError(headers={}, body={})
    monkeypatch.setattr(
        reranker_module.cohere, "ClientV2", lambda *, api_key, **_k: primary_client
    )

    with pytest.raises(TooManyRequestsError):
        CohereReranker().rerank("q", _candidates(1))


def test_fallback_is_not_used_when_primary_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reranker_module, "settings", _settings("primary", "fallback"))
    primary_client = Mock()
    primary_client.rerank.return_value = _rerank_response(1)
    fallback_client = Mock()
    monkeypatch.setattr(
        reranker_module.cohere,
        "ClientV2",
        lambda *, api_key, **_k: (
            primary_client if api_key == "primary" else fallback_client
        ),
    )

    CohereReranker().rerank("q", _candidates(1))

    fallback_client.rerank.assert_not_called()


def test_empty_fallback_secret_is_treated_as_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI sets COHERE_API_KEY_FALLBACK="" (empty, not unset - pydantic-settings
    parses this as SecretStr(''), never None) when the GitHub secret does not
    exist yet; that must not build a broken empty-key client."""
    settings = _settings("primary", None)
    settings.cohere_api_key_fallback = SecretStr("")
    monkeypatch.setattr(reranker_module, "settings", settings)
    primary_client = Mock()
    primary_client.rerank.side_effect = TooManyRequestsError(headers={}, body={})
    empty_key_client = Mock()  # would only be built if "" were wrongly accepted

    def fake_client_v2(*, api_key: str, **_kwargs: object) -> Mock:
        return primary_client if api_key == "primary" else empty_key_client

    monkeypatch.setattr(reranker_module.cohere, "ClientV2", fake_client_v2)

    with pytest.raises(TooManyRequestsError):
        CohereReranker().rerank("q", _candidates(1))

    empty_key_client.rerank.assert_not_called()
