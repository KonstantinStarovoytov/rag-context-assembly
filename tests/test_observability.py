from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import SecretStr

from src import observability
from src.config import Settings


def test_langfuse_credentials_are_optional_when_tracing_is_disabled(monkeypatch):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)

    configured = Settings(
        _env_file=None,
        openai_api_key="openai",
        cohere_api_key="cohere",
        tracing_enabled=False,
    )

    assert configured.langfuse_public_key is None
    assert configured.langfuse_secret_key is None


def test_get_langfuse_rejects_missing_credentials_when_tracing_is_enabled(monkeypatch):
    monkeypatch.setattr(
        observability,
        "settings",
        SimpleNamespace(
            langfuse_public_key=None,
            langfuse_secret_key=None,
            langfuse_base_url="https://cloud.langfuse.com",
        ),
    )
    observability.get_langfuse.cache_clear()

    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY.*LANGFUSE_SECRET_KEY"):
        observability.get_langfuse()


def test_traced_supports_readable_input_and_metadata_factories(monkeypatch):
    span = Mock()

    @contextmanager
    def observation(**kwargs):
        observation.kwargs = kwargs
        yield span

    client = SimpleNamespace(start_as_current_observation=observation)
    monkeypatch.setattr(
        observability,
        "settings",
        SimpleNamespace(tracing_enabled=True),
    )
    monkeypatch.setattr(observability, "get_langfuse", lambda: client)

    @observability.traced(
        "answer-question",
        "chain",
        input_factory=lambda query, **_: query,
        metadata_factory=lambda _query, **kwargs: kwargs,
        output_factory=lambda result: result["answer"],
    )
    def answer(query, *, strategy):
        return {"answer": "grounded"}

    assert answer("What is MCP?", strategy="hybrid") == {"answer": "grounded"}
    assert observation.kwargs == {
        "name": "answer-question",
        "as_type": "chain",
        "input": "What is MCP?",
        "metadata": {"strategy": "hybrid"},
    }
    span.update.assert_called_once_with(output="grounded")


def test_traced_marks_failed_observations_as_errors(monkeypatch):
    span = Mock()

    @contextmanager
    def observation(**_kwargs):
        yield span

    client = SimpleNamespace(start_as_current_observation=observation)
    monkeypatch.setattr(
        observability,
        "settings",
        SimpleNamespace(tracing_enabled=True),
    )
    monkeypatch.setattr(observability, "get_langfuse", lambda: client)

    @observability.traced("answer-question", "chain")
    def fail():
        raise RuntimeError("Qdrant is unavailable")

    with pytest.raises(RuntimeError, match="Qdrant is unavailable"):
        fail()

    span.update.assert_called_once_with(
        level="ERROR", status_message="Qdrant is unavailable"
    )


def test_model_config_uses_validated_langfuse_public_key(monkeypatch):
    handler = Mock()
    monkeypatch.setattr(
        observability,
        "settings",
        SimpleNamespace(
            tracing_enabled=True,
            langfuse_public_key="pk-test",
            langfuse_secret_key=SecretStr("sk-test"),
            langfuse_base_url="https://cloud.langfuse.com",
        ),
    )
    monkeypatch.setattr(observability, "get_langfuse", Mock())
    monkeypatch.setattr("langfuse.langchain.CallbackHandler", handler)

    config = observability.model_config("generate-answer")

    handler.assert_called_once_with(public_key="pk-test")
    assert config == {
        "callbacks": [handler.return_value],
        "run_name": "generate-answer",
    }
