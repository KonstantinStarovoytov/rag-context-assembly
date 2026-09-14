"""Bounded two-hop retrieval. Dependencies are injectable for offline evaluation."""

from collections.abc import Callable
from dataclasses import dataclass

from src.observability import traced
from src.rag.context_selector import select_generation_context
from src.rag.multi_query_retriever import _document_key
from src.rag.planner import RetrievalDecision, gap_query
from src.rag.retriever import SearchResult


@dataclass
class RetrievalOutcome:
    results: list
    rounds: int
    stop_reason: str
    decisions: list[RetrievalDecision]
    follow_up_queries: list[str]


@traced("two-hop-retrieval", "chain")
def retrieve_two_hop(
    question: str,
    initial_queries: list[str],
    *,
    retrieve: Callable,
    rerank: Callable,
    assess: Callable,
    context_k: int,
) -> RetrievalOutcome:
    if context_k < 1:
        raise ValueError("context_k must be positive")
    initial = retrieve(initial_queries)
    if not initial:
        return RetrievalOutcome([], 1, "no_initial_evidence", [], [])
    ranked = rerank(question, initial)
    selected = select_generation_context(ranked, context_k)
    decision = assess(question, selected)
    decisions = [decision]
    if decision.sufficient:
        return RetrievalOutcome(selected, 1, "sufficient", decisions, [])
    known_ids = {_document_key(r.document) for r in selected}
    seen_queries = {q.strip().casefold() for q in initial_queries}
    queries = []
    for gap in decision.gaps:
        if not set(gap.evidence_ids).issubset(known_ids):
            continue
        q = gap_query(question, gap)
        if q.casefold() in seen_queries:
            continue
        seen_queries.add(q.casefold())
        queries.append(q)
    if not queries:
        return RetrievalOutcome(selected, 1, "no_grounded_followups", decisions, [])
    # Exactly one additional retrieval round, regardless of the second assessment.
    followups = retrieve(queries[:3])
    documents = {_document_key(r.document): r for r in initial}
    new = [r for r in followups if _document_key(r.document) not in documents]
    if not new:
        return RetrievalOutcome(selected, 2, "no_new_documents", decisions, queries)
    for r in new:
        documents[_document_key(r.document)] = r
    # Re-rank the union against ORIGINAL intent; never compare scores across queries.
    union = [SearchResult(r.document, r.score) for r in documents.values()]
    selected = select_generation_context(rerank(question, union), context_k)
    final = assess(question, selected)
    decisions.append(final)
    return RetrievalOutcome(
        selected,
        2,
        "sufficient" if final.sufficient else "round_limit",
        decisions,
        queries,
    )
