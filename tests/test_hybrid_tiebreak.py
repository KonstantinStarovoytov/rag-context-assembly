from unittest.mock import Mock

from langchain_core.documents import Document

from src.rag import retriever
from src.rag.retriever import document_key, search_hybrid


def _doc(suffix: str) -> Document:
    return Document(
        page_content=f"chunk-{suffix}",
        metadata={"source": "https://docs.test", "h1": suffix},
    )


def test_hybrid_cut_is_stable_across_tied_qdrant_orders(monkeypatch):
    a, b, c = _doc("a"), _doc("b"), _doc("c")
    score = 0.166667
    # Three documents share one RRF score; only two slots are requested.
    orders = [
        [(a, score), (b, score), (c, score)],
        [(c, score), (a, score), (b, score)],
    ]
    store = Mock()
    store.similarity_search_with_score.side_effect = orders
    monkeypatch.setattr(retriever, "get_hybrid_vector_store", lambda: store)
    monkeypatch.setattr(retriever, "TIE_BREAK_OVERFETCH", 1)

    first = [document_key(r.document) for r in search_hybrid("q", k=2)]
    second = [document_key(r.document) for r in search_hybrid("q", k=2)]
    expected = sorted((a, b, c), key=document_key)[:2]
    assert first == second == [document_key(d) for d in expected]
