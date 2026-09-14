import argparse

from src.rag.reranker import CohereReranker
from src.rag.retriever import search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense retrieval with Cohere reranking",
    )

    parser.add_argument(
        "query",
        help="Search query",
    )

    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    print()
    print(f"Query: {args.query}")
    print()

    dense_results = search(
        query=args.query,
        k=args.retrieve_k,
    )

    print("DENSE RETRIEVAL")
    print("=" * 80)

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        metadata = result.document.metadata

        print(f"#{rank:<2} score={result.score:.4f} {metadata.get('title')}")

    reranker = CohereReranker()

    reranked = reranker.rerank(
        query=args.query,
        results=dense_results,
        top_n=args.top_n,
    )

    print()
    print("AFTER COHERE RERANK")
    print("=" * 80)

    for result in reranked:
        metadata = result.document.metadata

        heading = " > ".join(
            value
            for value in (
                metadata.get("h1"),
                metadata.get("h2"),
                metadata.get("h3"),
            )
            if value
        )

        print(
            f"#{result.rerank_rank:<2} "
            f"rerank={result.rerank_score:.4f} "
            f"dense_rank=#{result.original_rank:<2} "
            f"dense={result.retrieval_score:.4f}"
        )

        print(f"    {metadata.get('title')}")

        if heading:
            print(f"    {heading}")

        print()


if __name__ == "__main__":
    main()
