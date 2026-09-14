from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evals.seed_query_transform_dataset import INTENTS
from src.rag import pipeline
from src.rag.language import needs_english_translation
from src.rag.policy import (
    ENABLE_ITERATIVE_BY_DEFAULT,
    ENABLE_SEMANTIC_REWRITE,
    should_rewrite_after_miss,
)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("How do I configure Cursor rules only for Python files?", False),
        ("cursor rules python files only", False),
        ("как MCP server expose tools клиенту", True),
        ("как MCP server предоставляет tools клиенту", True),
        ("jak serwer MCP udostępnia tools klientowi", True),
    ],
)
def test_translation_gate_matches_rescore_failure_class(query, expected):
    assert needs_english_translation(query) is expected


def test_golden_set_non_english_variants_need_translation():
    for intent in INTENTS:
        assert needs_english_translation(intent["variants"]["ru"])
        assert needs_english_translation(intent["variants"]["pl"])
        assert not needs_english_translation(intent["variants"]["clean_en"])
        assert not needs_english_translation(intent["variants"]["short_en"])


def test_rescore_forbids_semantic_rewrite_and_default_iterative():
    assert ENABLE_SEMANTIC_REWRITE is False
    assert ENABLE_ITERATIVE_BY_DEFAULT is False
    assert should_rewrite_after_miss() is False


def test_hybrid_translates_only_non_english_queries(monkeypatch):
    hybrid = Mock(return_value=[object()])
    multi = Mock(return_value=[object()])
    translate = Mock(return_value="How does an MCP server expose tools?")
    monkeypatch.setattr(pipeline, "search_hybrid", hybrid)
    monkeypatch.setattr(pipeline, "search_queries_hybrid", multi)
    monkeypatch.setattr(pipeline, "translate_query", translate)
    monkeypatch.setattr(pipeline, "CohereReranker", Mock())
    monkeypatch.setattr(pipeline, "generate_answer", Mock(return_value="ok"))

    pipeline.answer_question(
        "How does an MCP server expose tools to clients?", strategy="hybrid"
    )
    assert hybrid.call_count == 1
    translate.assert_not_called()
    multi.assert_not_called()

    pipeline.answer_question(
        "как MCP server предоставляет tools клиенту", strategy="hybrid"
    )
    translate.assert_called_once()
    assert multi.call_args.kwargs["queries"][0] == (
        "как MCP server предоставляет tools клиенту"
    )
    assert hybrid.call_count == 1


def test_hybrid_english_still_translates_english(monkeypatch):
    multi = Mock(return_value=[object()])
    translate = Mock(return_value="same")
    monkeypatch.setattr(pipeline, "search_queries_hybrid", multi)
    monkeypatch.setattr(pipeline, "translate_query", translate)
    monkeypatch.setattr(
        pipeline,
        "CohereReranker",
        lambda: SimpleNamespace(rerank=Mock(return_value=[])),
    )
    monkeypatch.setattr(pipeline, "generate_answer", Mock(return_value="ok"))

    pipeline.answer_question("How do skills work?", strategy="hybrid-english")

    translate.assert_called_once_with("How do skills work?")
    assert multi.call_count == 1
