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


def test_index_snapshot_comes_from_qdrant_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "_read_index_meta", lambda: {"indexed_at": "2026-09-20"})
    monkeypatch.setattr(core.settings, "index_snapshot", "2026-01-01")
    core.forget_index_snapshot()

    assert core.index_snapshot() == "2026-09-20"


def test_index_snapshot_falls_back_to_env_when_meta_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> dict[str, Any]:
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(core, "_read_index_meta", boom)
    monkeypatch.setattr(core.settings, "index_snapshot", "2026-01-01")
    core.forget_index_snapshot()

    assert core.index_snapshot() == "2026-01-01"


def test_index_snapshot_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def read() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"indexed_at": "2026-09-20"}

    monkeypatch.setattr(core, "_read_index_meta", read)
    core.forget_index_snapshot()

    core.index_snapshot()
    core.index_snapshot()

    assert calls == 1


def test_search_passage_content_is_raw_not_the_embedding_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP search_docs promises passages are "quoted documentation text";
    page_content may carry the Vendor/Product/.../embedding prefix, which
    must not leak into what the calling agent sees."""
    from langchain_core.documents import Document

    from src.rag.retriever import SearchResult

    def fake_search_hybrid(*, query, k, vendor=None):
        return [
            SearchResult(
                document=Document(
                    page_content="Vendor: a\nProduct: b\nDocument: c\nSection: d\n\nreal text",
                    metadata={"raw_content": "real text", "source": "https://x"},
                ),
                score=0.9,
            )
        ]

    monkeypatch.setattr(core, "search_hybrid", fake_search_hybrid)

    passages = core.search("q")

    assert passages[0].content == "real text"
