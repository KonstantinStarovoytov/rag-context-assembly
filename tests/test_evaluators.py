from evals.evaluators import hit_at_10, reciprocal_rank


def test_hit_implies_nonzero_reciprocal_rank():
    expected = {"relevant": [{"title": "correct"}]}
    for rank in range(1, 11):
        output = {"results": [{"title": "wrong"}] * (rank - 1) + [{"title": "correct"}]}
        assert hit_at_10(output=output, expected_output=expected).value == 1
        assert (
            reciprocal_rank(output=output, expected_output=expected).value == 1 / rank
        )
