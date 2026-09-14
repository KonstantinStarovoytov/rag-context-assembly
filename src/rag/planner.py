"""Evidence assessment is separate from pre-retrieval translation."""

import json
from typing import Literal, cast

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator

from src.config import settings
from src.observability import model_config, prompt_context
from src.prompts.managed import get_chat_prompt
from src.rag.multi_query_retriever import _document_key
from src.rag.reranker import RerankResult

GapCategory = Literal[
    "when_to_use",
    "how_to_configure",
    "vendor_constraint",
    "comparison_aspect",
    "concept_definition",
]


class Gap(BaseModel):
    """A missing piece of evidence, not a search query.

    The judge only names the gap; `gap_query` turns it into retrieval terms.
    """

    category: GapCategory
    target: str = Field(min_length=1)
    slot: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class RetrievalDecision(BaseModel):
    sufficient: bool
    gaps: list[Gap] = Field(max_length=3)

    @model_validator(mode="after")
    def consistent(self) -> RetrievalDecision:
        if self.sufficient and self.gaps:
            raise ValueError("Sufficient evidence cannot have gaps")
        return self


def gap_query(question: str, gap: Gap) -> str:
    """Keep the original intent and append the gap, so sparse search keeps names."""
    return " ".join(f"{question} {gap.target} {gap.slot}".split())


class EvidencePlanner:
    def assess(self, question: str, results: list[RerankResult]) -> RetrievalDecision:
        evidence = [
            {
                "id": _document_key(r.document),
                "metadata": r.document.metadata,
                "text": r.document.page_content,
            }
            for r in results
        ]
        model = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            temperature=0,
        )
        prompt = get_chat_prompt(
            "evidence-planner",
            version=settings.evidence_planner_prompt_version,
            strict=settings.prompt_strict,
            payload=json.dumps(
                {"question": question, "evidence": evidence}, ensure_ascii=False
            ),
        )
        with prompt_context(prompt):
            return cast(
                RetrievalDecision,
                model.with_structured_output(RetrievalDecision).invoke(
                    prompt.messages,
                    config=model_config("assess-evidence"),
                ),
            )


def translate_query(query: str) -> str:
    model = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=0,
    )
    prompt = get_chat_prompt(
        "translate",
        version=settings.translate_prompt_version,
        strict=settings.prompt_strict,
        query=query,
    )
    with prompt_context(prompt):
        response = model.invoke(
            prompt.messages,
            config=model_config("translate-query"),
        )
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("Translator returned empty or non-text output")
    return response.content.strip()
