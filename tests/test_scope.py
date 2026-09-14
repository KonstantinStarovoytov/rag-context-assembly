import pytest
from langchain_core.documents import Document

from src.rag import retriever
from src.rag.scope import infer_single_vendor


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How do Cursor subagents work?", "cursor"),
        ("How should Claude Code package a plugin?", "anthropic"),
        ("How does Codex load AGENTS.md?", "openai"),
        ("How does an MCP server expose tools?", None),
        ("Compare Cursor and Claude skills", None),
    ],
)
def test_infer_single_vendor_is_conservative(query, expected):
    assert infer_single_vendor(query) == expected


def test_hybrid_search_applies_vendor_filter(monkeypatch):
    calls = []

    class FakeStore:
        def similarity_search_with_score(self, **kwargs):
            calls.append(kwargs)
            return [(Document(page_content="context", metadata={}), 0.5)]

    monkeypatch.setattr(retriever, "get_hybrid_vector_store", FakeStore)

    results = retriever.search_hybrid("Cursor MCP", k=7, vendor="cursor")

    assert len(results) == 1
    condition = calls[0]["filter"].must[0]
    assert condition.key == "metadata.vendor"
    assert condition.match.value == "cursor"
