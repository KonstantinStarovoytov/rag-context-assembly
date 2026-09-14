import argparse

from src.rag.retriever import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed agent documentation")

    parser.add_argument(
        "query",
        help="Search query",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results",
    )

    args = parser.parse_args()

    results = search(
        query=args.query,
        k=args.k,
    )

    print()
    print(f"Query: {args.query}")
    print()

    for index, result in enumerate(
        results,
        start=1,
    ):
        document = result.document
        metadata = document.metadata

        print("=" * 80)
        print(f"#{index}  score={result.score:.4f}")

        print(f"Vendor:  {metadata.get('vendor')}")

        print(f"Product: {metadata.get('product')}")

        print(f"Title:   {metadata.get('title')}")

        heading = " > ".join(
            value
            for key, value in (
                ("h1", metadata.get("h1")),
                ("h2", metadata.get("h2")),
                ("h3", metadata.get("h3")),
            )
            if value
        )

        if heading:
            print(f"Section: {heading}")

        print(f"Source:  {metadata.get('source')}")

        print()

        content = document.page_content

        print(content[:1000])

        if len(content) > 1000:
            print("...")

        print()


if __name__ == "__main__":
    main()
