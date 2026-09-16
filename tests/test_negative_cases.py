"""Questions the corpus cannot answer; the right answer is an abstention."""

from evals.negative_cases import CASES, DATASET_NAME, expected_output


def test_every_negative_case_expects_abstention() -> None:
    assert DATASET_NAME == "rag/negative-v1"
    assert len(CASES) >= 10
    assert len({case["id"] for case in CASES}) == len(CASES)
    for case in CASES:
        assert case["question"].strip()
        assert case["why_outside"].strip()
        assert expected_output(case) == {"expect_abstention": True}


def test_claude_desktop_connector_case_was_removed() -> None:
    """The MCP docs page documents claude_desktop_config.json directly, so
    this question was answerable from the corpus; it was not a negative case."""
    assert "claude-desktop-connectors" not in {case["id"] for case in CASES}
