"""Tests for the Sultan- / SimAlign-style contextual word aligners.

Marked `slow` because the first call downloads XLM-R (~1.1 GB)."""
import pytest

from ryokai import Ryokai


@pytest.mark.slow
@pytest.mark.parametrize("aligner", ["hungarian", "argmax", "itermax", "mai"])
def test_contextual_aligner_self_score_is_one_en(aligner):
    scorer = Ryokai(aligner=aligner)
    s = scorer.score(
        reference="The cat sat on the mat.",
        hypothesis="The cat sat on the mat.",
        target_lang="en",
    )
    assert s.f1 > 0.95


@pytest.mark.slow
@pytest.mark.parametrize("aligner", ["hungarian", "argmax", "itermax", "mai"])
def test_contextual_aligner_paraphrase_beats_unrelated_en(aligner):
    scorer = Ryokai(aligner=aligner)
    near = scorer.score(
        reference="The cat sat on the mat.",
        hypothesis="A cat is sitting on the mat.",
        target_lang="en",
    ).f1
    far = scorer.score(
        reference="The cat sat on the mat.",
        hypothesis="Tomorrow the rocket flies to Mars.",
        target_lang="en",
    ).f1
    assert near > far


@pytest.mark.slow
def test_contextual_aligner_threshold_drops_weak_alignments():
    pair = ("The cat sat on the mat.", "Cars drive fast.")
    f_lo = Ryokai(aligner="itermax", threshold=0.0).score(
        reference=pair[0], hypothesis=pair[1], target_lang="en",
    ).f1
    f_hi = Ryokai(aligner="itermax", threshold=0.95).score(
        reference=pair[0], hypothesis=pair[1], target_lang="en",
    ).f1
    assert f_hi <= f_lo


@pytest.mark.slow
def test_itermax_recall_at_least_argmax():
    pair = ("The cat sat on the mat yesterday.", "A cat is sitting on the mat.")
    s_arg = Ryokai(aligner="argmax").score(
        reference=pair[0], hypothesis=pair[1], target_lang="en",
    )
    s_iter = Ryokai(aligner="itermax").score(
        reference=pair[0], hypothesis=pair[1], target_lang="en",
    )
    assert s_iter.recall >= s_arg.recall - 1e-6


@pytest.mark.slow
def test_contextual_aligner_content_only_works():
    scorer = Ryokai(aligner="itermax", content_only=True)
    s = scorer.score(
        reference="The cat sat on the mat.",
        hypothesis="A cat is on the mat.",
        target_lang="en",
    )
    assert 0.0 <= s.f1 <= 1.0
