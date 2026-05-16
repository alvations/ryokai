"""End-to-end tests for all 13 MEANT-supported languages.

Marked `slow` because each language triggers HF model downloads on first
run (one shared multilingual XLM-R POS model + one shared multilingual
embedding model). Run with: `pytest -m slow`.
"""
import pytest

from pymeant import MEANT, SUPPORTED_LANGS


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
def test_meant_self_score_is_one(lang):
    """Scoring a sentence against itself should be ~1.0."""
    ref, _, _ = CASES[lang]
    meant = MEANT(lang=lang)
    score = meant.score(ref, ref)
    assert score.f1 > 0.95, f"self-score for {lang} = {score.f1:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGS))
def test_meant_paraphrase_beats_unrelated(lang):
    """A near-paraphrase should outscore an unrelated sentence."""
    ref, near, far = CASES[lang]
    meant = MEANT(lang=lang)
    s_near = meant.score(ref, near).f1
    s_far = meant.score(ref, far).f1
    assert s_near > s_far, (
        f"{lang}: paraphrase {s_near:.3f} did not beat unrelated {s_far:.3f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGS))
def test_meant_score_in_unit_interval(lang):
    """All scores must be in [0, 1]."""
    ref, near, far = CASES[lang]
    meant = MEANT(lang=lang)
    for hyp in (ref, near, far):
        s = meant.score(ref, hyp)
        assert 0.0 <= s.f1 <= 1.0
        assert 0.0 <= s.precision <= 1.0
        assert 0.0 <= s.recall <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGS))
def test_nosrl_self_score_is_one(lang):
    """The YiSi-1-style no-SRL fallback also works for all 13 langs."""
    ref, _, _ = CASES[lang]
    meant = MEANT(lang=lang, use_srl=False)
    score = meant.score(ref, ref)
    assert score.f1 > 0.99


@pytest.mark.slow
def test_xmeant_reference_free_cross_lingual():
    """XMEANT mode: score Chinese MT against English source (or vice versa)."""
    meant = MEANT(lang="zh")  # target = zh
    # source = English sentence; hypothesis = its Chinese translation
    src = "The cat sat on the mat yesterday."
    good = "猫昨天坐在垫子上。"
    bad = "明天火箭将飞向火星。"
    s_good = meant.score_xlingual(src, good, source_lang="en").f1
    s_bad = meant.score_xlingual(src, bad, source_lang="en").f1
    assert s_good > s_bad
