from types import SimpleNamespace

from evals import answer_quality


def test_quality_metrics_fan_out_one_judge_assessment(monkeypatch):
    assessment = answer_quality.AnswerQualityAssessment(
        faithfulness=0.75,
        answer_relevance=0.5,
        context_utilization=1.0,
        citation_correctness=0.25,
        unsupported_claims=["Unsupported detail"],
        missing_aspects=["Authorization"],
        citation_issues=["[2] does not support the claim"],
        summary="One claim is unsupported.",
    )
    monkeypatch.setattr(
        answer_quality,
        "assess_answer_quality",
        lambda *_args: answer_quality.AnswerQualityResult(
            assessment=assessment,
            prompt_metadata={"prompt_name": "doc-bot/answer-evaluator"},
        ),
    )
    monkeypatch.setattr(
        answer_quality,
        "settings",
        SimpleNamespace(openai_chat_model="judge-model"),
    )

    scores = answer_quality.answer_quality_metrics(
        input={"question": "How are tools authorized?"},
        output={
            "answer": "An answer [1].",
            "contexts": [{"citation": 1, "content": "Context"}],
        },
    )

    assert [(score.name, score.value) for score in scores] == [
        ("answer_faithfulness", 0.75),
        ("answer_relevance", 0.5),
        ("context_utilization", 1.0),
        ("citation_correctness", 0.25),
    ]
    assert scores[0].metadata["unsupported_claims"] == ["Unsupported detail"]


def test_missing_answer_or_context_scores_zero_without_calling_judge(monkeypatch):
    judge = SimpleNamespace()
    monkeypatch.setattr(answer_quality, "assess_answer_quality", judge)

    scores = answer_quality.answer_quality_metrics(
        input={"question": "Question"}, output={"answer": "", "contexts": []}
    )

    assert [score.value for score in scores] == [0.0, 0.0, 0.0, 0.0]
