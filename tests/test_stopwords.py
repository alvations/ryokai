"""Tests for the bundled multilingual stopword loader."""
from ryokai import SUPPORTED_LANGS
from ryokai.stopwords import is_content_token, stopwords_for


def test_stopwords_loaded_for_all_13_meant_languages():
    for lang in SUPPORTED_LANGS:
        sw = stopwords_for(lang)
        assert len(sw) > 0, f"empty stopwords for {lang}"


def test_english_stopwords_include_articles():
    sw = stopwords_for("en")
    for w in ("the", "a", "an", "of", "and"):
        assert w in sw, f"{w!r} missing from English stopwords"


def test_chinese_stopwords_include_de():
    sw = stopwords_for("zh")
    assert "的" in sw


def test_is_content_token_filters_stopwords_and_punctuation():
    assert not is_content_token("the", "en")
    assert not is_content_token("THE", "en")  # case-insensitive
    assert not is_content_token(".", "en")
    assert not is_content_token("...", "en")
    assert not is_content_token(",", "en")
    assert not is_content_token("", "en")
    # content words should pass
    assert is_content_token("cat", "en")
    assert is_content_token("anarchism", "en")


def test_unknown_lang_returns_empty():
    assert stopwords_for("xx") == set()
    # unknown lang -> nothing is a stopword, only punctuation gets filtered
    assert is_content_token("hello", "xx")
    assert not is_content_token("...", "xx")
