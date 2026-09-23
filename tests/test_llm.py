"""GPT-6 models reject any temperature but the default, so it must be optional."""

import pytest

from src.rag import llm


def test_chat_model_omits_temperature_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm.settings, "openai_temperature", None)

    params = llm.chat_model()._default_params

    assert "temperature" not in params or params["temperature"] is None


def test_chat_model_passes_explicit_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm.settings, "openai_temperature", 0.0)

    assert llm.chat_model().temperature == 0.0


def test_empty_temperature_env_means_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import Settings

    monkeypatch.setenv("OPENAI_TEMPERATURE", "")

    assert Settings().openai_temperature is None
