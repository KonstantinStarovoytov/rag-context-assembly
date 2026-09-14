"""Compare standard hybrid retrieval with bounded two-hop retrieval coverage."""

import time

from langfuse import Langfuse

from evals.evidence_coverage import (
    DATASET_NAME,
    complete_evidence_coverage,
    evidence_coverage,
)
from src.config import settings
from src.rag.context_selector import select_generation_context
from src.rag.iterative import retrieve_two_hop
from src.rag.multi_query_retriever import search_queries_hybrid
from src.rag.planner import EvidencePlanner
from src.rag.reranker import CohereReranker
from src.rag.retriever import SearchResult, search_hybrid

RETRIEVAL_TOP_K = 10
GENERATION_TOP_K = settings.generation_top_k
PER_QUERY_TOP_K = 20
COHERE_REQUEST_INTERVAL_SECONDS = 6.5

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key.get_secret_value(),
    base_url=settings.langfuse_base_url,
)


def _result_row(result: SearchResult) -> dict:
    metadata = result.document.metadata
    return {
        "rank": result.rerank_rank,
        "vendor": metadata.get("vendor"),
        "product": metadata.get("product"),
        "title": metadata.get("title"),
        "heading": " > ".join(
            value
            for value in (metadata.get("h1"), metadata.get("h2"), metadata.get("h3"))
            if value
        ),
        "source": metadata.get("source"),
        "retrieval_score": result.retrieval_score,
        "rerank_score": result.rerank_score,
    }


def _rerank(question: str, candidates: list[SearchResult]) -> list[SearchResult]:
    ranked = CohereReranker().rerank(question, candidates, top_n=RETRIEVAL_TOP_K)
    time.sleep(COHERE_REQUEST_INTERVAL_SECONDS)
    return ranked


def baseline_task(*, item, **_kwargs) -> dict:
    question = item.input["question"]
    ranked = _rerank(question, search_hybrid(question, k=RETRIEVAL_TOP_K))
    selected = select_generation_context(ranked, GENERATION_TOP_K)
    return {
        "results": [_result_row(result) for result in selected],
        "rounds": 1,
        "stop_reason": "single_pass",
        "follow_up_queries": [],
    }


def iterative_task(*, item, **_kwargs) -> dict:
    question = item.input["question"]
    outcome = retrieve_two_hop(
        question,
        [question],
        retrieve=lambda queries: search_queries_hybrid(
            queries, per_query_k=PER_QUERY_TOP_K, final_k=RETRIEVAL_TOP_K
        ),
        rerank=_rerank,
        assess=EvidencePlanner().assess,
        context_k=GENERATION_TOP_K,
    )
    return {
        "results": [_result_row(result) for result in outcome.results],
        "rounds": outcome.rounds,
        "stop_reason": outcome.stop_reason,
        "follow_up_queries": outcome.follow_up_queries,
        "planner_sufficient": [decision.sufficient for decision in outcome.decisions],
    }


def _metadata(mode: str) -> dict:
    return {
        "mode": mode,
        "retriever": "Qdrant hybrid dense+BM25 RRF",
        "reranker": settings.cohere_rerank_model,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "generation_top_k": GENERATION_TOP_K,
        "per_query_top_k": PER_QUERY_TOP_K if mode == "iterative" else None,
        "answer_prompt_version": settings.answer_prompt_version or "production:v1",
        "translate_prompt_version": settings.translate_prompt_version
        or "production:v1",
        "planner_prompt_version": (
            settings.evidence_planner_prompt_version or "production:v1"
        ),
    }


def main() -> None:
    dataset = langfuse.get_dataset(DATASET_NAME)
    common = {
        "evaluators": [evidence_coverage, complete_evidence_coverage],
        "max_concurrency": 1,
    }
    baseline = dataset.run_experiment(
        name="evidence-coverage-baseline-v3-section-diverse",
        description="Single-pass hybrid retrieval, Cohere rerank, and section-diverse final top-5 evidence coverage.",
        task=baseline_task,
        metadata=_metadata("baseline"),
        **common,
    )
    print(baseline.format())
    iterative = dataset.run_experiment(
        name="evidence-coverage-iterative-v3-section-diverse",
        description="Bounded two-hop hybrid retrieval and section-diverse final top-5 evidence coverage.",
        task=iterative_task,
        metadata=_metadata("iterative"),
        **common,
    )
    print(iterative.format())
    langfuse.flush()


if __name__ == "__main__":
    main()
