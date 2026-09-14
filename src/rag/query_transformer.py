from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.observability import model_config
from src.prompts import load_prompt

SYSTEM_PROMPT = load_prompt("query-transform")


class QueryTransformResult(BaseModel):
    normalized: str = Field(
        description=(
            "Cleaned-up version of the query in the original language. "
            "Same meaning, no new information."
        )
    )

    english: str = Field(
        description=(
            "Faithful English translation of the original query. Do not expand it."
        )
    )

    semantic_variant: str = Field(
        description=(
            "Alternative English formulation expressing exactly "
            "the same information need."
        )
    )


class QueryTransformer:
    def __init__(self) -> None:
        model = ChatOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_chat_model,
            temperature=0,
        )

        self.model = model.with_structured_output(QueryTransformResult)

    def build_queries(
        self,
        query: str,
    ) -> RetrievalQueries:
        transformed = self.transform(query)

        return RetrievalQueries(
            original=query,
            normalized=transformed.normalized,
            english=transformed.english,
            semantic_variant=(transformed.semantic_variant),
        )

    def transform(
        self,
        query: str,
    ) -> QueryTransformResult:
        return self.model.invoke(
            [
                (
                    "system",
                    SYSTEM_PROMPT,
                ),
                (
                    "human",
                    query,
                ),
            ],
            config=model_config("transform-query"),
        )


class RetrievalQueries(BaseModel):
    original: str
    normalized: str
    english: str
    semantic_variant: str

    def unique_queries(self) -> list[str]:
        queries = [
            self.original,
            self.english,
            self.semantic_variant,
        ]

        unique = []
        seen = set()

        for query in queries:
            value = query.strip()
            key = value.casefold()

            if not value or key in seen:
                continue

            seen.add(key)
            unique.append(value)

        return unique
