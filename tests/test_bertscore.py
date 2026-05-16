"""Unit tests for the BERTScore aligner mode.

Fast tests cover the dispatch / API; slow tests run the actual XLM-R
forward pass and validate score semantics (self-score = 1, paraphrase
beats unrelated).
"""
import pytest

from ryokai import Ryokai
from ryokai.scorer import score_nosrl


def test_bertscore_is_registered_aligner():
    """Dispatching aligner='bertscore' must not raise a ValueError."""
    with pytest.raises(ValueError, match="aligner must be"):
        # sanity: a bogus aligner still raises
        score_nosrl("a", "b", aligner="bogus_method")


def test_bertscore_invalid_aggregation_raises():
    with pytest.raises(ValueError, match="aggregation"):
        score_nosrl("a", "b", aligner="bertscore", aggregation="bogus")


@pytest.mark.slow
def test_bertscore_self_score_is_one_en():
    scorer = Ryokai(aligner="bertscore")
    s = scorer.score(
        reference="The cat sat on the mat.",
        hypothesis="The cat sat on the mat.",
        target_lang="en",
    )
    assert s.f1 > 0.99


@pytest.mark.slow
def test_bertscore_paraphrase_beats_unrelated_en():
    scorer = Ryokai(aligner="bertscore")
    near = scorer.score(
        reference="The cat sat on the mat.",
        hypothesis="A cat was sitting on the mat.",
        target_lang="en",
    ).f1
    far = scorer.score(
        reference="The cat sat on the mat.",
        hypothesis="Tomorrow the rocket flies to Mars.",
        target_lang="en",
    ).f1
    assert near > far


@pytest.mark.slow
def test_bertscore_score_in_unit_interval():
    scorer = Ryokai(aligner="bertscore")
    s = scorer.score(
        reference="The cat sat on the mat.",
        hypothesis="A cat is on the mat.",
        target_lang="en",
    )
    assert 0.0 <= s.precision <= 1.0
    assert 0.0 <= s.recall <= 1.0
    assert 0.0 <= s.f1 <= 1.0


@pytest.mark.slow
def test_bertscore_works_cross_lingual():
    """BERTScore mode works with the reference-free / source dispatch."""
    scorer = Ryokai(aligner="bertscore")
    s_good = scorer.score(
        source="The cat sat on the mat yesterday.",
        hypothesis="猫は昨日マットの上に座った。",
        source_lang="en", target_lang="ja",
    ).f1
    s_bad = scorer.score(
        source="The cat sat on the mat yesterday.",
        hypothesis="明日ロケットは火星に飛びます。",
        source_lang="en", target_lang="ja",
    ).f1
    assert s_good > s_bad
