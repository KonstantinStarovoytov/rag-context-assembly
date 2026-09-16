"""_tool_error: the message must match the cause, because the model acts on it.

A single "retry in a few seconds" was wrong for two real cases: a Cohere
monthly quota (retrying burns OpenAI calls for translate/embedding that are
already paid by the time rerank fails) and a suspended Qdrant cluster (the
owner has to act, no retry helps).
"""

import pytest
from cohere.errors.too_many_requests_error import TooManyRequestsError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.service import core
from src.service.mcp_server import _tool_error


def test_rejected_question_keeps_its_reason() -> None:
    assert str(_tool_error(core.QuestionRejected("Question must not be empty"))) == (
        "Question must not be empty"
    )


def test_cohere_quota_does_not_advise_a_retry() -> None:
    message = str(_tool_error(TooManyRequestsError(headers={}, body={})))

    assert "rerank" in message.lower() and "quota" in message.lower()
    assert "retry in a few seconds" not in message


@pytest.mark.parametrize(
    "error",
    [
        ResponseHandlingException(ConnectionError("refused")),
        UnexpectedResponse(
            status_code=503, reason_phrase="down", content=b"", headers=None
        ),
    ],
)
def test_qdrant_outage_says_the_index_needs_the_owner(error: Exception) -> None:
    message = str(_tool_error(error))

    assert "index" in message.lower()
    assert "owner" in message.lower()
    assert "retry in a few seconds" not in message


def test_openai_rate_limit_advises_a_short_retry() -> None:
    import httpx2 as httpx
    import openai

    response = httpx.Response(429, request=httpx.Request("POST", "https://x"))
    error = openai.RateLimitError("slow down", response=response, body=None)

    message = str(_tool_error(error))

    assert "rate limit" in message.lower()
    assert "retry" in message.lower()


def test_unknown_failure_keeps_the_generic_retry_advice() -> None:
    message = str(_tool_error(RuntimeError("openai down")))

    assert "temporarily unavailable" in message and "retry" in message
    assert "openai down" not in message
