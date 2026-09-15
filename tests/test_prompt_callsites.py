from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from src.prompts.managed import ManagedPrompt
from src.rag import generator, planner
from src.rag.reranker import RerankResult


def _managed(name, messages):
    return ManagedPrompt(
        name=name,
        messages=messages,
        version=2,
        source="langfuse",
        content_hash="hash",
        langfuse_prompt=object(),
    )


def test_generator_uses_versioned_answer_prompt_and_links_it(monkeypatch):
    prompt = _managed(
        "doc-bot/answer",
        [
            {"role": "system", "content": "grounding"},
            {"role": "user", "content": "question and context"},
        ],
    )
    get_prompt = Mock(return_value=prompt)
    entered = []

    @contextmanager
    def prompt_context(value):
        entered.append(value)
        yield

    model = Mock()
    model.invoke.return_value = SimpleNamespace(content="answer [1]")
    monkeypatch.setattr(generator, "get_chat_prompt", get_prompt)
    monkeypatch.setattr(generator, "prompt_context", prompt_context)
    monkeypatch.setattr(generator, "chat_model", Mock(return_value=model))
    result = RerankResult(
        document=Document(
            page_content="context",
            metadata={"title": "Doc", "source": "https://example.test"},
        ),
        retrieval_score=0.2,
        rerank_score=0.8,
        original_rank=1,
        rerank_rank=1,
    )

    generated = generator.generate_from_selected("question", [result])

    get_prompt.assert_called_once_with(
        "answer",
        version=generator.settings.answer_prompt_version,
        strict=generator.settings.prompt_strict,
        query="question",
        context=generator._format_context([result]),
    )
    assert entered == [prompt]
    assert model.invoke.call_args.args[0] == prompt.messages
    assert generated.answer == "answer [1]"


def test_generate_from_selected_does_not_drop_same_section_chunks(monkeypatch):
    prompt = _managed(
        "doc-bot/answer",
        [
            {"role": "system", "content": "grounding"},
            {"role": "user", "content": "question and context"},
        ],
    )
    monkeypatch.setattr(generator, "get_chat_prompt", Mock(return_value=prompt))
    monkeypatch.setattr(generator, "chat_model", Mock())
    monkeypatch.setattr(
        generator,
        "prompt_context",
        lambda _prompt: __import__("contextlib").nullcontext(),
    )
    generator.chat_model.return_value.invoke.return_value = SimpleNamespace(
        content="ok"
    )
    chunks = [
        RerankResult(
            Document(
                page_content=f"step {i}",
                metadata={"source": "https://docs.test/mcp.md", "h1": "Install"},
            ),
            1,
            1,
            i,
            i,
        )
        for i in range(1, 4)
    ]
    generated = generator.generate_from_selected("how to install", chunks)
    assert [source.citation for source in generated.sources] == [1, 2, 3]
    context = generator.get_chat_prompt.call_args.kwargs["context"]
    assert "step 1" in context and "step 2" in context and "step 3" in context


def test_translate_uses_versioned_prompt_and_links_it(monkeypatch):
    prompt = _managed(
        "doc-bot/translate",
        [
            {"role": "system", "content": "translate"},
            {"role": "user", "content": "вопрос"},
        ],
    )
    monkeypatch.setattr(planner, "get_chat_prompt", Mock(return_value=prompt))
    linked = Mock()

    @contextmanager
    def prompt_context(value):
        linked(value)
        yield

    model = Mock()
    model.invoke.return_value = SimpleNamespace(content="question")
    monkeypatch.setattr(planner, "prompt_context", prompt_context)
    monkeypatch.setattr(planner, "chat_model", Mock(return_value=model))

    assert planner.translate_query("вопрос") == "question"
    linked.assert_called_once_with(prompt)
    assert model.invoke.call_args.args[0] == prompt.messages


def test_evidence_planner_uses_pinned_prompt_settings(monkeypatch):
    prompt = _managed(
        "doc-bot/evidence-planner",
        [
            {"role": "system", "content": "assess"},
            {"role": "user", "content": "payload"},
        ],
    )
    get_prompt = Mock(return_value=prompt)
    monkeypatch.setattr(planner, "get_chat_prompt", get_prompt)

    structured = Mock()
    structured.invoke.return_value = SimpleNamespace()
    model = Mock()
    model.with_structured_output.return_value = structured
    monkeypatch.setattr(planner, "chat_model", Mock(return_value=model))

    planner.EvidencePlanner().assess("question", [])

    get_prompt.assert_called_once()
    assert get_prompt.call_args.kwargs["version"] == (
        planner.settings.evidence_planner_prompt_version
    )
    assert get_prompt.call_args.kwargs["strict"] == planner.settings.prompt_strict
    assert structured.invoke.call_args.args[0] == prompt.messages
