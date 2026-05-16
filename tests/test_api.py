"""Lightweight API surface tests."""
import pytest

from ryokai import MEANT, SUPPORTED_LANGS


def test_supported_langs_is_13_languages():
    assert len(SUPPORTED_LANGS) == 13
    assert set(SUPPORTED_LANGS) == {
        "en", "de", "fr", "es", "cs", "fi", "hi",
        "lv", "pl", "ro", "ru", "tr", "zh",
    }


def test_meant_init_validates_lang():
    with pytest.raises(ValueError):
        MEANT(lang="qq", use_srl=False)


def test_meant_xlingual_validates_source_lang():
    m = MEANT(lang="en", use_srl=False)
    with pytest.raises(ValueError):
        m.score_xlingual("hola", "hello", source_lang="zz")


def test_score_corpus_length_mismatch_raises():
    m = MEANT(lang="en", use_srl=False)
    with pytest.raises(ValueError):
        m.score_corpus(["a"], ["b", "c"])
