"""One place that builds the chat model, so every call site is bounded alike."""

from langchain_openai import ChatOpenAI

from src.config import settings

# The OpenAI SDK defaults to 600 s and two retries: a hung request would hold
# a worker thread for half an hour. Generation over 8 chunks finishes well
# inside a minute; one retry covers a transient 5xx without doubling the bill.
CHAT_TIMEOUT_SECONDS = 60.0
EMBED_TIMEOUT_SECONDS = 30.0


def chat_model() -> ChatOpenAI:
    # Pinned like every other call site, but this alone does not make answers
    # reproducible: repeated runs over an identical context still differ.
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=0,
        timeout=CHAT_TIMEOUT_SECONDS,
        max_retries=1,
    )
