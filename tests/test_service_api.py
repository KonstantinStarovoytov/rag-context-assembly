from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.service import app as app_module
from src.service import auth, core

TOKEN = "test-token"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(auth, "required_token", lambda: TOKEN)
    monkeypatch.setattr(app_module, "trace_url", lambda: None)
    monkeypatch.setattr(app_module, "flush", lambda: None)
    with TestClient(app_module.create_app()) as test_client:
        yield test_client


def _answer(*_args: Any, **_kwargs: Any) -> core.Answer:
    return core.Answer(
        answer="Skills load on demand [1].",
        sources=[
            core.Source(
                citation=1,
                title="Skills",
                heading="Overview",
                url="https://docs.test/skills.md",
                rerank_score=0.91,
            )
        ],
        strategy="hybrid",
        iterative=False,
    )


def test_health_needs_no_token(client: Any) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_without_token_is_rejected(client: Any) -> None:
    response = client.post("/ask", json={"question": "How do skills work?"})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_ask_with_wrong_token_is_rejected(client: Any) -> None:
    response = client.post(
        "/ask",
        json={"question": "How do skills work?"},
        headers={"Authorization": "Bearer nope"},
    )

    assert response.status_code == 401


def test_ask_returns_answer_and_sources(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core, "answer", _answer)

    response = client.post(
        "/ask",
        json={"question": "How do skills work?"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Skills load on demand [1]."
    assert payload["sources"][0]["url"] == "https://docs.test/skills.md"
    assert payload["strategy"] == "hybrid"


def test_overlong_question_is_rejected_before_retrieval(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fail(*_args: Any, **_kwargs: Any) -> core.Answer:
        nonlocal called
        called = True
        raise AssertionError("pipeline must not run")

    monkeypatch.setattr(core, "answer_question", fail)
    monkeypatch.setattr(app_module.settings, "api_max_question_chars", 10)

    response = client.post(
        "/ask",
        json={"question": "x" * 11},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 422
    assert called is False


def test_mcp_endpoint_requires_token(client: Any) -> None:
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Accept": "application/json, text/event-stream"},
    )

    assert response.status_code == 401
