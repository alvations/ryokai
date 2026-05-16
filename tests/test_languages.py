"""End-to-end tests for all 13 supported languages.

Marked `slow` because the first run downloads HF models (one shared
multilingual XLM-R POS model + one shared multilingual embedding model).
Run with: `pytest -m slow`.
"""
import pytest

from ryokai import Ryokai, SUPPORTED_LANGS


# (lang, reference, near-paraphrase, unrelated)
CASES = {
    "en": (
        "The cat sat on the mat yesterday.",
        "A cat was sitting on the mat yesterday.",
        "Tomorrow the rocket will fly to Mars.",
    ),
    "de": (
        "Die Katze saß gestern auf der Matte.",
        "Eine Katze hat gestern auf der Matte gesessen.",
        "Morgen fliegt die Rakete zum Mars.",
    ),
    "fr": (
        "Le chat s'est assis sur le tapis hier.",
        "Un chat était assis hier sur le tapis.",
        "Demain la fusée volera vers Mars.",
    ),
    "es": (
        "El gato se sentó ayer en la alfombra.",
        "Un gato estuvo sentado ayer en la alfombra.",
        "Mañana el cohete volará hacia Marte.",
    ),
    "cs": (
        "Kočka včera seděla na rohoži.",
        "Včera seděla na rohoži kočka.",
        "Zítra raketa poletí na Mars.",
    ),
    "fi": (
        "Kissa istui matolla eilen.",
        "Eilen kissa istui matolla.",
        "Huomenna raketti lentää Marsiin.",
    ),
    "hi": (
        "बिल्ली कल चटाई पर बैठी थी।",
        "कल बिल्ली चटाई पर बैठी।",
        "कल रॉकेट मंगल ग्रह पर जाएगा।",
    ),
    "lv": (
        "Kaķis vakar sēdēja uz paklāja.",
        "Vakar kaķis sēdēja uz paklāja.",
        "Rīt raķete lidos uz Marsu.",
    ),
    "pl": (
        "Kot wczoraj siedział na macie.",
        "Wczoraj kot siedział na macie.",
        "Jutro rakieta poleci na Marsa.",
    ),
    "ro": (
        "Pisica a stat ieri pe covor.",
        "Ieri pisica a stat pe covor.",
        "Mâine racheta va zbura spre Marte.",
    ),
    "ru": (
        "Кошка вчера сидела на коврике.",
        "Вчера кошка сидела на коврике.",
        "Завтра ракета полетит на Марс.",
    ),
    "tr": (
        "Kedi dün paspasın üzerinde oturdu.",
        "Dün kedi paspasın üzerinde oturuyordu.",
        "Yarın roket Mars'a uçacak.",
    ),
    "zh": (
        "猫昨天坐在垫子上。",
        "昨天猫坐在了垫子上。",
        "明天火箭将飞向火星。",
    ),
}


def test_supported_langs_match_test_cases():
    assert set(SUPPORTED_LANGS) == set(CASES)


@pytest.mark.slow
@pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGS))
def test_nosrl_self_score_is_one(lang):
    ref, _, _ = CASES[lang]
    scorer = Ryokai()
    s = scorer.score(reference=ref, hypothesis=ref, target_lang=lang)
    assert s.f1 > 0.99


@pytest.mark.slow
@pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGS))
def test_srl_self_score_is_one(lang):
    ref, _, _ = CASES[lang]
    scorer = Ryokai()
    s = scorer.score(reference=ref, hypothesis=ref, target_lang=lang, srl=True)
    assert s.f1 > 0.95


@pytest.mark.slow
@pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGS))
def test_paraphrase_beats_unrelated_nosrl(lang):
    ref, near, far = CASES[lang]
    scorer = Ryokai()
    s_near = scorer.score(reference=ref, hypothesis=near, target_lang=lang).f1
    s_far = scorer.score(reference=ref, hypothesis=far, target_lang=lang).f1
    assert s_near > s_far


@pytest.mark.slow
@pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGS))
def test_paraphrase_beats_unrelated_srl(lang):
    ref, near, far = CASES[lang]
    scorer = Ryokai()
    s_near = scorer.score(reference=ref, hypothesis=near, target_lang=lang, srl=True).f1
    s_far = scorer.score(reference=ref, hypothesis=far, target_lang=lang, srl=True).f1
    assert s_near > s_far


@pytest.mark.slow
@pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGS))
def test_score_in_unit_interval(lang):
    ref, near, far = CASES[lang]
    scorer = Ryokai()
    for hyp in (ref, near, far):
        s = scorer.score(reference=ref, hypothesis=hyp, target_lang=lang)
        assert 0.0 <= s.f1 <= 1.0
        assert 0.0 <= s.precision <= 1.0
        assert 0.0 <= s.recall <= 1.0


@pytest.mark.slow
def test_reference_free_cross_lingual_nosrl():
    """Reference-free path: score Chinese MT against English source."""
    scorer = Ryokai()
    src = "The cat sat on the mat yesterday."
    good = "猫昨天坐在垫子上。"
    bad = "明天火箭将飞向火星。"
    s_good = scorer.score(
        source=src, hypothesis=good,
        source_lang="en", target_lang="zh",
    ).f1
    s_bad = scorer.score(
        source=src, hypothesis=bad,
        source_lang="en", target_lang="zh",
    ).f1
    assert s_good > s_bad


@pytest.mark.slow
def test_reference_free_cross_lingual_srl():
    """XMEANT proper: source-vs-hyp frame-based."""
    scorer = Ryokai()
    src = "The cat sat on the mat yesterday."
    good = "猫昨天坐在垫子上。"
    bad = "明天火箭将飞向火星。"
    s_good = scorer.score(
        source=src, hypothesis=good,
        source_lang="en", target_lang="zh", srl=True,
    ).f1
    s_bad = scorer.score(
        source=src, hypothesis=bad,
        source_lang="en", target_lang="zh", srl=True,
    ).f1
    assert s_good > s_bad
