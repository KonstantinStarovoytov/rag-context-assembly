"""Questions the indexed documentation does not answer.

The corpus is 35 pages on customising Claude Code, Cursor, Codex and MCP. An
agent that has this MCP server connected will also ask about everything
around it; the correct answer then is "the documentation does not cover
this", and until now no eval measured whether the system says so. Each case
records why it is outside the corpus so a reviewer can disagree.
"""

from typing import Any

DATASET_NAME = "rag/negative-v1"

CASES: list[dict[str, str]] = [
    {
        "id": "copilot-custom-instructions",
        "question": "How do I write custom instructions for GitHub Copilot in VS Code?",
        "why_outside": "GitHub Copilot is not one of the indexed products.",
    },
    {
        "id": "windsurf-cascade-rules",
        "question": "Where do Windsurf Cascade rules live and how are they scoped?",
        "why_outside": "Windsurf is not an indexed product.",
    },
    {
        "id": "claude-api-pricing",
        "question": "What does Claude Opus cost per million input tokens on the API?",
        "why_outside": "Claude API pricing is outside the Claude Code docs subset.",
    },
    {
        "id": "claude-messages-api-streaming",
        "question": "How do I stream a response from the Claude Messages API in Python?",
        "why_outside": "Messages API reference is not indexed; only Claude Code pages are.",
    },
    {
        "id": "cursor-subscription-price",
        "question": "How much does a Cursor Pro subscription cost per month?",
        "why_outside": "Cursor pricing pages are not in the include list.",
    },
    {
        "id": "python-decorator",
        "question": "How do I write a decorator in Python that caches function results?",
        "why_outside": "General programming; no product documentation involved.",
    },
    {
        "id": "openai-fine-tuning",
        "question": "How do I fine-tune an OpenAI model on my own dataset?",
        "why_outside": "OpenAI platform fine-tuning docs are not indexed; only Codex pages.",
    },
    {
        "id": "langchain-retriever",
        "question": "How do I build a hybrid retriever with LangChain and Qdrant?",
        "why_outside": "LangChain and Qdrant docs are not part of the corpus.",
    },
    {
        "id": "mcp-spec-2024-transport",
        "question": "What did the 2024-11-05 MCP revision say about HTTP+SSE transport?",
        "why_outside": "Only the newest MCP docs version is indexed; older revisions are not.",
    },
    {
        "id": "jetbrains-ai-assistant-mcp",
        "question": "How do I add an MCP server to JetBrains AI Assistant?",
        "why_outside": "JetBrains client docs are not indexed.",
    },
    {
        "id": "claude-desktop-connectors",
        "question": "How do I add a custom connector in the Claude desktop app settings?",
        "why_outside": "claude.ai / Claude Desktop help centre is not indexed.",
    },
    {
        "id": "kubernetes-deploy",
        "question": "How do I deploy an MCP server on Kubernetes with a Helm chart?",
        "why_outside": "Deployment tooling outside the spec and vendor docs.",
    },
]


def expected_output(case: dict[str, str]) -> dict[str, Any]:
    return {"expect_abstention": True}
