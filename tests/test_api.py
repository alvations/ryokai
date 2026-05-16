"""Lightweight API surface tests for the keyword-dispatch interface."""
import pytest

from ryokai import Ryokai, SUPPORTED_LANGS


def test_supported_langs_is_13_languages():
    assert len(SUPPORTED_LANGS) == 13
    assert set(SUPPORTED_LANGS) == {
        "en", "de", "fr", "es", "cs", "fi", "hi",
        "lv", "pl", "ro", "ru", "tr", "zh",
    }


def test_score_requires_source_xor_reference():
    scorer = Ryokai()
    with pytest.raises(ValueError):
        scorer.score(hypothesis="hi", target_lang="en")
    with pytest.raises(ValueError):
        scorer.score(
            hypothesis="hi", target_lang="en",
            source="hola", reference="hi",
        )


def test_score_validates_target_lang():
    scorer = Ryokai()
    with pytest.raises(ValueError):
        scorer.score(reference="x", hypothesis="y", target_lang="qq")


def test_score_validates_source_lang_when_given():
    scorer = Ryokai()
    with pytest.raises(ValueError):
        scorer.score(
            source="x", hypothesis="y",
            target_lang="en", source_lang="zz",
        )


def test_score_corpus_requires_sources_xor_references():
    scorer = Ryokai()
    with pytest.raises(ValueError):
        scorer.score_corpus(hypotheses=["a"], target_lang="en")
    with pytest.raises(ValueError):
        scorer.score_corpus(
            hypotheses=["a"], target_lang="en",
            sources=["b"], references=["c"],
        )


def test_score_corpus_length_mismatch_raises():
    scorer = Ryokai()
    with pytest.raises(ValueError):
        scorer.score_corpus(
            hypotheses=["a"], target_lang="en",
            references=["b", "c"],
        )


def test_meant_alias_is_ryokai():
    from ryokai import MEANT
    assert MEANT is Ryokai
