"""Tests for the YiSi-1-style no-SRL fallback scorer."""
import numpy as np

from pymeant.scorer import score_nosrl


class _IdentitySim:
    """1.0 iff strings match exactly, else 0.0. Deterministic stub."""

    def sim(self, a, b):
        out = np.zeros((len(a), len(b)), dtype=np.float32)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x == y:
                    out[i, j] = 1.0
        return out


def test_identical_tokens_score_one():
    s = score_nosrl("the cat sat", "the cat sat", sim_backend=_IdentitySim())
    assert s.f1 == 1.0


def test_disjoint_tokens_score_zero():
    s = score_nosrl("alpha beta", "gamma delta", sim_backend=_IdentitySim())
    assert s.f1 == 0.0


def test_partial_overlap_between_zero_and_one():
    s = score_nosrl("the cat sat", "the dog sat", sim_backend=_IdentitySim())
    assert 0.0 < s.f1 < 1.0


def test_both_empty_is_perfect_match():
    s = score_nosrl("", "", sim_backend=_IdentitySim())
    assert s.f1 == 1.0


def test_one_empty_is_zero():
    s = score_nosrl("", "hello", sim_backend=_IdentitySim())
    assert s.f1 == 0.0
    s = score_nosrl("hello", "", sim_backend=_IdentitySim())
    assert s.f1 == 0.0


def test_extra_hyp_tokens_drops_precision():
    s_baseline = score_nosrl("a b", "a b", sim_backend=_IdentitySim())
    s_extra = score_nosrl("a b", "a b c d", sim_backend=_IdentitySim())
    assert s_extra.precision < s_baseline.precision
    assert s_extra.recall == s_baseline.recall
