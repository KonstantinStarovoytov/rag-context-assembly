from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from src import observability
from src.rag import query_transformer, retriever
from src.rag.query_transformer import QueryTransformer
from src.rag.reranker import CohereReranker
from src.rag.retriever import SearchResult


def _capture_observations(monkeypatch):
    observations = []

    @contextmanager
    def start(**kwargs):
        span = Mock()
        observations.append((kwargs, span))
        yield span

    monkeypatch.setattr(
        observability,
        "settings",
        SimpleNamespace(tracing_enabled=True),
    )
    monkeypatch.setattr(
        observability,
        "get_langfuse",
        lambda: SimpleNamespace(
            start_as_current_observation=start, get_current_trace_id=lambda: None
        ),
    )
    return observations


def test_dense_search_is_traced_as_retriever(monkeypatch):
    observations = _capture_observations(monkeypatch)
    store = Mock()
    store.similarity_search_with_score.return_value = []
    monkeypatch.setattr(retriever, "get_vector_store", lambda: store)

    assert retriever.search("MCP tools", k=7) == []

    assert observations[0][0] == {
        "name": "dense-search",
        "as_type": "retriever",
        "input": {"query": "MCP tools", "k": 7},
    }


def test_reranker_trace_contains_query_candidates_and_ranked_output(monkeypatch):
    observations = _capture_observations(monkeypatch)
    document = Document(
        page_content="Tools are exposed by servers.",
        metadata={"source": "https://example.test/mcp", "title": "MCP"},
    )
    candidate = SearchResult(document=document, score=0.25)
    reranker = CohereReranker.__new__(CohereReranker)
    reranker.model = "rerank-test"
    reranker.client = Mock()
    reranker.client.rerank.return_value = SimpleNamespace(
        results=[SimpleNamespace(index=0, relevance_score=0.9)]
    )

    ranked = reranker.rerank("How are tools exposed?", [candidate], top_n=1)

    trace_input = observations[0][0]["input"]
    assert observations[0][0]["name"] == "rerank-candidates"
    assert observations[0][0]["as_type"] == "retriever"
    assert trace_input["query"] == "How are tools exposed?"
    assert trace_input["candidates"][0]["source"] == "https://example.test/mcp"
    observations[0][1].update.assert_called_once()
    trace_output = observations[0][1].update.call_args.kwargs["output"]
    assert trace_output[0]["original_rank"] == 1
    assert trace_output[0]["rerank_rank"] == 1
    assert ranked[0].rerank_score == 0.9


def test_query_transformer_passes_langfuse_callback_config(monkeypatch):
    transformer = QueryTransformer.__new__(QueryTransformer)
    transformer.model = Mock()
    expected = {"callbacks": [object()]}
    config = Mock(return_value=expected)
    monkeypatch.setattr(query_transformer, "model_config", config)

    transformer.transform("question")

    config.assert_called_once_with("transform-query")
    assert transformer.model.invoke.call_args.kwargs["config"] is expected
