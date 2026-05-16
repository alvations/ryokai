"""Unit tests for the AER / precision / recall / F1 word-alignment metrics."""
from ryokai.eval import aer, f1, precision, recall


def test_aer_perfect_match_is_zero():
    A = {(0, 0), (1, 1), (2, 2)}
    S = {(0, 0), (1, 1), (2, 2)}
    assert aer(A, S) == 0.0


def test_aer_completely_wrong_is_one():
    A = {(0, 1), (1, 2), (2, 0)}
    S = {(0, 0), (1, 1), (2, 2)}
    assert aer(A, S) == 1.0


def test_aer_uses_possible_set_for_partial_credit():
    A = {(0, 0), (1, 1)}
    S = {(0, 0)}                # 1 sure alignment
    P = {(0, 0), (1, 1)}        # (1,1) is permissible but not sure
    # |A ∩ S| = 1, |A ∩ P| = 2 -> num = 3; |A| + |S| = 2 + 1 = 3 -> AER = 0
    assert aer(A, S, P) == 0.0


def test_aer_empty_predicted_is_one_when_gold_nonempty():
    A: set = set()
    S = {(0, 0)}
    assert aer(A, S) == 1.0


def test_aer_both_empty_is_zero():
    assert aer(set(), set()) == 0.0


def test_aer_accepts_3tuples_from_ryokai_aligners():
    """ryokai aligners return (i, j, similarity); AER should accept those."""
    pairs_with_scores = [(0, 0, 0.9), (1, 1, 0.85)]
    S = {(0, 0), (1, 1)}
    assert aer(pairs_with_scores, S) == 0.0


def test_precision_recall_f1_basic():
    A = {(0, 0), (1, 1), (2, 3)}
    G = {(0, 0), (1, 1), (2, 2)}
    assert precision(A, G) == 2 / 3
    assert recall(A, G) == 2 / 3
    assert abs(f1(A, G) - 2 / 3) < 1e-9


def test_evaluate_aligner_returns_aggregate():
    """Wire-test using a tiny stub aligner — no model download."""
    class _StubAligner:
        def align(self, ref, hyp, method="itermax", threshold=0.5):
            # always predict identity alignment of length min(ref, hyp)
            n = min(len(ref.split()), len(hyp.split()))
            return [(i, i, 1.0) for i in range(n)], ref.split(), hyp.split()

    from ryokai.eval import evaluate_aligner
    res = evaluate_aligner(
        _StubAligner(),
        [
            ("a b c", "a b c", {(0, 0), (1, 1), (2, 2)}, None),
            ("x y", "x y", {(0, 0), (1, 1)}, None),
        ],
    )
    assert res["n"] == 2
    assert res["aer"] == 0.0
    assert res["f1"] == 1.0
