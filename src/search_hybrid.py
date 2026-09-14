import argparse

from src.rag.retriever import search_hybrid


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        type=str,
    )

    args = parser.parse_args()

    results = search_hybrid(
        args.query,
        k=10,
    )

    print(f"\nQuery: {args.query}\n")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        document = result.document
        metadata = document.metadata

        headings = [
            metadata.get("h1"),
            metadata.get("h2"),
            metadata.get("h3"),
        ]

        heading = " > ".join(value for value in headings if value)

        print("=" * 80)

        print(f"#{rank} score={result.score:.4f}")

        print(f"Vendor: {metadata.get('vendor')}")

        print(f"Product: {metadata.get('product')}")

        print(f"Title: {metadata.get('title')}")

        print(f"Section: {heading}")

        print(f"Source: {metadata.get('source')}")

        print()


if __name__ == "__main__":
    main()
