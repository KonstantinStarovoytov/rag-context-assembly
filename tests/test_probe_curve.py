"""Retrieval-only coverage curve over the candidate pool size."""

from typing import Any

from evals.probe_context_assembly import coverage_curve

CASE = {
    "id": "c",
    "question": "q",
    "evidence_groups": [
        {"id": "g1", "target": {"vendor": "cursor", "source_contains": "rules"}},
        {"id": "g2", "target": {"vendor": "cursor", "source_contains": "hooks"}},
    ],
}


class _Doc:
    def __init__(self, source: str) -> None:
        self.metadata = {"vendor": "cursor", "source": source, "title": "t"}
        self.page_content = ""


class _Ranked:
    def __init__(self, source: str) -> None:
        self.document = _Doc(source)


def test_curve_reports_complete_coverage_per_pool_size() -> None:
    # The second required source only appears at rank 6.
    ranked = [_Ranked("rules"), *[_Ranked("other")] * 4, _Ranked("hooks")]

    def retrieve_and_rank(question: str, pool: int) -> list[Any]:
        return ranked[:pool]

    curve = coverage_curve([CASE], [5, 10], retrieve_and_rank)

    assert curve == {5: 0.0, 10: 1.0}
