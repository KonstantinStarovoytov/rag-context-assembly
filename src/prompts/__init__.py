"""Local prompt assets; kept stable for reproducible experiments."""

from importlib.resources import files


def load_prompt(name: str) -> str:
    if name not in {
        "answer",
        "answer-evaluator",
        "answer-v1",
        "query-transform",
        "translate",
        "evidence-planner",
    }:
        raise ValueError(f"Unknown prompt: {name}")
    return (
        files(__package__).joinpath(f"{name}.txt").read_text(encoding="utf-8").strip()
    )
