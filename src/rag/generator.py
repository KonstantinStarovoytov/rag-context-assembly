from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from src.config import settings
from src.observability import model_config, prompt_context
from src.prompts.managed import get_chat_prompt
from src.rag.context_selector import select_generation_context
from src.rag.reranker import RerankResult


@dataclass(frozen=True, slots=True)
class AnswerSource:
    citation: int
    title: str
    heading: str
    url: str
    rerank_score: float


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    sources: list[AnswerSource]


def _get_heading(result: RerankResult) -> str:
    metadata = result.document.metadata

    headings = [
        metadata.get("h1"),
        metadata.get("h2"),
        metadata.get("h3"),
    ]

    return " > ".join(heading for heading in headings if heading)


def _format_context(
    results: list[RerankResult],
) -> str:
    parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        metadata = result.document.metadata

        title = metadata.get(
            "title",
            "Untitled",
        )

        heading = _get_heading(result)

        source = metadata.get(
            "source",
            "",
        )

        parts.append(
            f"""
[{index}]
Title: {title}
Section: {heading}
Source: {source}

{result.document.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(parts)


def generate_from_selected(
    query: str,
    selected: list[RerankResult],
) -> GeneratedAnswer:
    """Generate from an already chosen context. Does not re-run the selector."""
    if not selected:
        return GeneratedAnswer(
            answer=("I couldn't find relevant documentation for this question."),
            sources=[],
        )

    context = _format_context(selected)
    prompt = get_chat_prompt(
        "answer",
        version=settings.answer_prompt_version,
        strict=settings.prompt_strict,
        query=query,
        context=context,
    )

    # Pinned like every other call site, but this alone does not make answers
    # reproducible: repeated runs over an identical context still differ.
    model = ChatOpenAI(
        api_key=(settings.openai_api_key.get_secret_value()),
        model=settings.openai_chat_model,
        temperature=0,
    )

    with prompt_context(prompt):
        response = model.invoke(
            prompt.messages,
            config=model_config("generate-answer"),
        )

    if isinstance(response.content, str):
        answer = response.content
    else:
        answer = str(response.content)

    sources = []

    for index, result in enumerate(
        selected,
        start=1,
    ):
        metadata = result.document.metadata

        sources.append(
            AnswerSource(
                citation=index,
                title=metadata.get(
                    "title",
                    "Untitled",
                ),
                heading=_get_heading(result),
                url=metadata.get(
                    "source",
                    "",
                ),
                rerank_score=(result.rerank_score),
            )
        )

    return GeneratedAnswer(
        answer=answer,
        sources=sources,
    )


def generate_answer(
    query: str,
    results: list[RerankResult],
    top_k: int | None = None,
) -> GeneratedAnswer:
    if top_k is None:
        top_k = settings.generation_top_k
    return generate_from_selected(query, select_generation_context(results, top_k))
