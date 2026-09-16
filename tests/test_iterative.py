from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from src.config import settings
from src.rag.iterative import retrieve_two_hop
from src.rag.multi_query_retriever import _document_key
from src.rag.planner import Gap, RetrievalDecision, gap_query
from src.rag.retriever import SearchResult


import pytest as _pytest


@_pytest.fixture(autouse=True)
def _no_relevance_floor(monkeypatch):
    """This module tests round orchestration (stop conditions, dedup), not the
    min_rerank_score threshold (covered by test_context_selector.py); its fake
    rerank passes SearchResult through unchanged, which has no rerank_score."""
    monkeypatch.setattr(settings, "min_rerank_score", None)


def result(text):
    return SearchResult(
        Document(page_content=text, metadata={"source": "https://docs.test"}), 1
    )


def gap(evidence_ids, slot="activation rule"):
    return Gap(
        category="when_to_use",
        target="Skill",
        slot=slot,
        evidence_ids=evidence_ids,
    )


def decision(sufficient=False, gaps=None):
    return RetrievalDecision(sufficient=sufficient, gaps=gaps or [])


def run(retrieve, assess):
    rerank = Mock(side_effect=lambda q, rows: rows)
    out = retrieve_two_hop(
        "original",
        ["original"],
        retrieve=retrieve,
        rerank=rerank,
        assess=assess,
        context_k=5,
    )
    assert all(c.args[0] == "original" for c in rerank.call_args_list)
    return out


def test_sufficient_stops_without_followups():
    retrieve = Mock(return_value=[result("one")])
    out = run(retrieve, Mock(return_value=decision(True)))
    assert out.stop_reason == "sufficient"
    assert retrieve.call_count == 1


def test_gap_query_keeps_question_and_gap():
    q = gap_query("When is a Skill loaded?", gap(["id"]))
    assert q == "When is a Skill loaded? Skill activation rule"


def test_gap_query_is_deterministic_and_normalized():
    g = gap(["id"], slot=" activation  rule ")
    assert gap_query(" original ", g) == gap_query("original", gap(["id"]))


@pytest.mark.parametrize("mode", ["unknown-id", "repeat-query", "none"])
def test_rejects_unusable_gaps(mode):
    r = result("one")
    known = _document_key(r.document)
    g = gap(["bad"] if mode == "unknown-id" else [known])
    retrieve = Mock(return_value=[r])
    initial = gap_query("original", g) if mode == "repeat-query" else "original"
    rerank = Mock(side_effect=lambda q, rows: rows)
    out = retrieve_two_hop(
        "original",
        [initial],
        retrieve=retrieve,
        rerank=rerank,
        assess=Mock(return_value=decision(gaps=[] if mode == "none" else [g])),
        context_k=5,
    )
    assert out.stop_reason == "no_grounded_followups"
    assert retrieve.call_count == 1


def test_round_limit_and_dedup():
    a, b = result("one"), result("two")
    g = gap([_document_key(a.document)])
    retrieve = Mock(side_effect=[[a], [a, b, b]])
    assess = Mock(return_value=decision(gaps=[g]))
    out = run(retrieve, assess)
    assert out.rounds == 2 and out.stop_reason == "round_limit"
    assert out.follow_up_queries == [gap_query("original", g)]
    assert len(out.results) == 2
    assert retrieve.call_count == 2 and assess.call_count == 2


def test_no_new_evidence_stops():
    a = result("one")
    retrieve = Mock(return_value=[a])
    assess = Mock(return_value=decision(gaps=[gap([_document_key(a.document)])]))
    assert run(retrieve, assess).stop_reason == "no_new_documents"
    assert assess.call_count == 1


def test_no_initial_evidence():
    assess = Mock(side_effect=AssertionError("must not assess empty evidence"))
    assert run(Mock(return_value=[]), assess).stop_reason == "no_initial_evidence"


def test_rejects_over_budget_plan():
    with pytest.raises(ValidationError):
        decision(gaps=[gap(["id"])] * 4)


def test_rejects_unknown_gap_category():
    with pytest.raises(ValidationError):
        Gap(category="bridge_entity", target="Skill", slot="x", evidence_ids=["id"])
