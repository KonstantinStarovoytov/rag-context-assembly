import argparse

from src.rag.query_transformer import (
    QueryTransformer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")

    args = parser.parse_args()

    transformer = QueryTransformer()

    result = transformer.build_queries(args.query)

    print()
    print("ORIGINAL")
    print(result.original)

    print()
    print("NORMALIZED")
    print(result.normalized)

    print()
    print("ENGLISH")
    print(result.english)

    print()
    print("SEMANTIC VARIANT")
    print(result.semantic_variant)

    print()
    print("RETRIEVAL QUERIES")

    for index, query in enumerate(
        result.unique_queries(),
        start=1,
    ):
        print(f"{index}. {query}")


if __name__ == "__main__":
    main()
