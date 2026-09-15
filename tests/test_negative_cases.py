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
