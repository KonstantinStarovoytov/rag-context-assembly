from langfuse import Langfuse

from src.config import settings

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key.get_secret_value(),
    base_url=settings.langfuse_base_url,
)

DATASET_NAME = "rag/retrieval-baseline"


CASES = [
    {
        "question": "How can Claude subagents preload skills?",
        "expected_title": "Create custom subagents",
        "expected_heading": "Control subagent capabilities",
    },
    {
        "question": "How does Claude Code connect to external tools?",
        "expected_title": "Connect Claude Code to tools via MCP",
        "expected_heading": None,
    },
    {
        "question": "How can I package reusable instructions for Claude?",
        "expected_title": "Extend Claude with skills",
        "expected_heading": None,
    },
]


def main() -> None:
    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Retrieval evaluation dataset for agent documentation RAG",
    )

    for case in CASES:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={
                "question": case["question"],
            },
            expected_output={
                "title": case["expected_title"],
                "heading": case["expected_heading"],
            },
        )


if __name__ == "__main__":
    main()
