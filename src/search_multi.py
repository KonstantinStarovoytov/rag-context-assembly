import argparse

from src.rag.multi_query_retriever import (
    search_multi_query_hybrid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")

    args = parser.parse_args()

    results = search_multi_query_hybrid(
        query=args.query,
        per_query_k=20,
        final_k=10,
    )

    print()
    print(f"Query: {args.query}")
    print()

    for rank, result in enumerate(
        results,
        start=1,
    ):
        metadata = result.document.metadata

        headings = [
            metadata.get("h1"),
            metadata.get("h2"),
            metadata.get("h3"),
        ]

        heading = " > ".join(value for value in headings if value)

        print("=" * 80)
        print(f"#{rank} RRF={result.score:.5f}")
        print(f"Vendor: {metadata.get('vendor')}")
        print(f"Product: {metadata.get('product')}")
        print(f"Title: {metadata.get('title')}")
        print(f"Section: {heading}")
        print()


if __name__ == "__main__":
    main()
