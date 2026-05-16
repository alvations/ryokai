"""Tests for the YiSi-1 / WOLVESAAR-style no-SRL fallback scorer."""
import numpy as np
import pytest

from ryokai.scorer import score_nosrl


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


def test_content_only_drops_stopwords():
    # With content_only=False, "the the the cat" has 4 tokens
    # With content_only=True, only "cat" remains
    s_with_stop = score_nosrl(
        "the the the cat", "a a a dog",
        sim_backend=_IdentitySim(),
        content_only=False, lang="en",
    )
    s_no_stop = score_nosrl(
        "the the the cat", "a a a dog",
        sim_backend=_IdentitySim(),
        content_only=True, lang="en",
    )
    # without stop filter: 0 of 4 align (cat != dog)
    assert s_with_stop.f1 == 0.0
    # with stop filter: cat vs dog -> still 0 align, but we computed on content words only
    assert s_no_stop.f1 == 0.0
    # different test: identical content words but different stopwords
    s = score_nosrl(
        "the cat", "a cat",
        sim_backend=_IdentitySim(),
        content_only=True, lang="en",
    )
    assert s.f1 == 1.0  # both reduce to just "cat"


def test_aggregation_harmonic_equals_f1_mathematically():
    s_f1 = score_nosrl("the cat sat", "the dog sat",
                       sim_backend=_IdentitySim(), aggregation="f1")
    s_h = score_nosrl("the cat sat", "the dog sat",
                      sim_backend=_IdentitySim(), aggregation="harmonic")
    assert abs(s_f1.f1 - s_h.f1) < 1e-9


def test_invalid_aggregation_raises():
    with pytest.raises(ValueError):
        score_nosrl("a", "b", sim_backend=_IdentitySim(), aggregation="bogus")


def test_invalid_aligner_raises():
    with pytest.raises(ValueError):
        score_nosrl("a", "b", sim_backend=_IdentitySim(), aligner="bogus")
