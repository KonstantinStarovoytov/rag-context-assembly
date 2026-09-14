import argparse

from src.observability import flush, trace_url
from src.rag.pipeline import answer_question


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
        type=str,
    )

    parser.add_argument(
        "--strategy", choices=["dense", "hybrid", "hybrid-english"], default=None
    )

    parser.add_argument("--iterative", action="store_true")
    args = parser.parse_args()

    try:
        result = answer_question(
            args.question, strategy=args.strategy, iterative=args.iterative
        )
    finally:
        flush()
        url = trace_url()
        if url:
            print(f"TRACE: {url}")

    print()
    print("ANSWER")
    print("=" * 80)
    print(result.answer)

    print()
    print("SOURCES")
    print("=" * 80)

    for source in result.sources:
        print(f"[{source.citation}] {source.title}")

        if source.heading:
            print(f"    {source.heading}")

        print(f"    {source.url}")

        print(f"    rerank_score={source.rerank_score:.4f}")

        print()


if __name__ == "__main__":
    try:
        main()
    finally:
        flush()
