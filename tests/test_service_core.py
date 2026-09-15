"""The shared handler forwards what both transports need, nothing more."""

from typing import Any

import pytest

from src.service import core


def test_search_forwards_vendor_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_hybrid(*, query: str, k: int, vendor: str | None = None) -> list[Any]:
        seen.update(query=query, k=k, vendor=vendor)
        return []

    monkeypatch.setattr(core, "search_hybrid", fake_hybrid)

    core.search("rules", limit=3, vendor="openai")

    assert seen == {"query": "rules", "k": 3, "vendor": "openai"}
