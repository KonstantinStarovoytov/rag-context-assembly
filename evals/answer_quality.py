"""Strict, traceable answer-quality metrics for the curated RAG dataset.

Ragas' current package requires downgrading this application's OpenAI SDK, so this
module keeps the same core evaluation questions in the application's compatible
LangChain/OpenAI stack. The scores are LLM-as-a-judge signals, not human ground
truth; the structured issue lists make each score auditable in Langfuse.
"""

import json
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from langfuse import Evaluation
from pydantic import BaseModel, Field

from src.config import settings
from src.observability import model_config, prompt_context, traced
from src.prompts.managed import get_chat_prompt


class AnswerQualityAssessment(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    context_utilization: float = Field(ge=0.0, le=1.0)
    citation_correctness: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str]
    missing_aspects: list[str]
    citation_issues: list[str]
    summary: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class AnswerQualityResult:
    assessment: AnswerQualityAssessment
    prompt_metadata: dict[str, str | int | None]


def _assessment_trace_input(
    question: str, answer: str, contexts: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }


@traced(
    "evaluate-answer-quality",
    "evaluator",
    input_factory=_assessment_trace_input,
)
def assess_answer_quality(
    question: str, answer: str, contexts: list[dict[str, Any]]
) -> AnswerQualityResult:
    """Judge one answer once, then fan its structured scores into Langfuse."""
    prompt = get_chat_prompt(
        "answer-evaluator",
        version=settings.answer_evaluator_prompt_version,
        # An experiment must never silently compare a remote production prompt
        # against a local fallback.
        strict=True,
        question=question,
        answer=answer,
        contexts=json.dumps(contexts, ensure_ascii=False),
    )
    model = ChatOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_chat_model,
        temperature=0,
    )
    with prompt_context(prompt):
        assessment = model.with_structured_output(AnswerQualityAssessment).invoke(
            prompt.messages,
            config=model_config("evaluate-answer-quality"),
        )
    if not isinstance(assessment, AnswerQualityAssessment):
        assessment = AnswerQualityAssessment.model_validate(assessment)
    return AnswerQualityResult(
        assessment=assessment,
        prompt_metadata=prompt.metadata,
    )


def answer_quality_metrics(*, input, output, **_kwargs) -> list[Evaluation]:
    """Return all quality dimensions from one judge invocation."""
    answer = output.get("answer", "")
    contexts = output.get("contexts", [])
    if not answer or not contexts:
        return [
            Evaluation(
                name=name,
                value=0.0,
                comment="No generated answer or selected context was available.",
            )
            for name in (
                "answer_faithfulness",
                "answer_relevance",
                "context_utilization",
                "citation_correctness",
            )
        ]

    quality = assess_answer_quality(input["question"], answer, contexts)
    assessment = quality.assessment
    issues = {
        "unsupported_claims": assessment.unsupported_claims,
        "missing_aspects": assessment.missing_aspects,
        "citation_issues": assessment.citation_issues,
        "summary": assessment.summary,
        "evaluator_model": settings.openai_chat_model,
        **quality.prompt_metadata,
    }
    return [
        Evaluation(
            name="answer_faithfulness",
            value=assessment.faithfulness,
            comment=assessment.summary,
            metadata=issues,
        ),
        Evaluation(
            name="answer_relevance",
            value=assessment.answer_relevance,
            comment=f"missing_aspects={assessment.missing_aspects}",
            metadata=issues,
        ),
        Evaluation(
            name="context_utilization",
            value=assessment.context_utilization,
            comment="Selected contexts were judged against the generated answer.",
            metadata=issues,
        ),
        Evaluation(
            name="citation_correctness",
            value=assessment.citation_correctness,
            comment=f"citation_issues={assessment.citation_issues}",
            metadata=issues,
        ),
    ]
