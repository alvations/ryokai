"""Tests for the Sultan- / SimAlign-style contextual word aligners.

Marked `slow` because the first call downloads XLM-R (~1.1 GB)."""
import pytest

from ryokai import MEANT


@pytest.mark.slow
@pytest.mark.parametrize("aligner", ["hungarian", "argmax", "itermax"])
def test_contextual_aligner_self_score_is_one_en(aligner):
    m = MEANT(lang="en", use_srl=False, aligner=aligner)
    s = m.score("The cat sat on the mat.", "The cat sat on the mat.")
    assert s.f1 > 0.95


@pytest.mark.slow
@pytest.mark.parametrize("aligner", ["hungarian", "argmax", "itermax"])
def test_contextual_aligner_paraphrase_beats_unrelated_en(aligner):
    m = MEANT(lang="en", use_srl=False, aligner=aligner)
    near = m.score("The cat sat on the mat.", "A cat is sitting on the mat.").f1
    far = m.score("The cat sat on the mat.", "Tomorrow the rocket flies to Mars.").f1
    assert near > far


@pytest.mark.slow
def test_contextual_aligner_threshold_drops_weak_alignments():
    m_lo = MEANT(lang="en", use_srl=False, aligner="itermax", threshold=0.0)
    m_hi = MEANT(lang="en", use_srl=False, aligner="itermax", threshold=0.95)
    pair = ("The cat sat on the mat.", "Cars drive fast.")
    f_lo = m_lo.score(*pair).f1
    f_hi = m_hi.score(*pair).f1
    assert f_hi <= f_lo


@pytest.mark.slow
def test_itermax_recall_at_least_argmax():
    """IterMax adds back unaligned words -> recall >= ArgMax."""
    pair = ("The cat sat on the mat yesterday.", "A cat is sitting on the mat.")
    s_arg = MEANT(lang="en", use_srl=False, aligner="argmax").score(*pair)
    s_iter = MEANT(lang="en", use_srl=False, aligner="itermax").score(*pair)
    assert s_iter.recall >= s_arg.recall - 1e-6


@pytest.mark.slow
def test_contextual_aligner_content_only_works():
    m = MEANT(lang="en", use_srl=False, aligner="itermax", content_only=True)
    s = m.score("The cat sat on the mat.", "A cat is on the mat.")
    assert 0.0 <= s.f1 <= 1.0
