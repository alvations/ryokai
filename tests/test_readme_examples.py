"""Run the exact code samples from the README's Quickstart and Variants
sections against real models. If these tests fail, the README is lying.

All marked `slow` because they download the multilingual MiniLM and
(when srl=True) the XLM-R UDPOS model and (for bertscore /
contextual aligners) XLM-R itself.
"""
import pytest

from ryokai import Ryokai, EmbeddingSimBackend


# Exactly the strings appearing in README.md
SRC = "The cat sat on the mat yesterday."
HYP = "猫は昨日マットの上に座った。"
REF = "猫は昨日マットの上に座った。"
PARA = "猫は昨日マットの上に座っていた。"
HYP_BAD = "明日ロケットは火星に飛びます。"


@pytest.mark.slow
def test_readme_quickstart():
    """Quickstart sample — verbatim, default-model kwarg included."""
    scorer = Ryokai(sim_backend=EmbeddingSimBackend("minilm"))
    score = scorer.score(
        source=SRC, hypothesis=HYP,
        source_lang="en", target_lang="ja",
    )
    assert 0.0 <= score.f1 <= 1.0


@pytest.mark.slow
def test_readme_variant_1_ref_free_alignment_embedding():
    scorer = Ryokai(sim_backend=EmbeddingSimBackend("minilm"))
    s = scorer.score(
        source=SRC, hypothesis=HYP,
        source_lang="en", target_lang="ja",
    )
    assert 0.0 <= s.f1 <= 1.0


@pytest.mark.slow
def test_readme_variant_2_ref_based_alignment_embedding():
    scorer = Ryokai(sim_backend=EmbeddingSimBackend("minilm"))
    s_para = scorer.score(reference=REF, hypothesis=PARA, target_lang="ja")
    s_bad = scorer.score(reference=REF, hypothesis=HYP_BAD, target_lang="ja")
    assert s_para.f1 > s_bad.f1, (
        f"paraphrase {s_para.f1:.3f} did not beat unrelated {s_bad.f1:.3f}"
    )


@pytest.mark.slow
def test_readme_variant_3_ref_free_srl_xmeant():
    scorer = Ryokai(sim_backend=EmbeddingSimBackend("minilm"))
    s_good = scorer.score(
        source=SRC, hypothesis=HYP,
        source_lang="en", target_lang="ja", srl=True,
    )
    s_bad = scorer.score(
        source=SRC, hypothesis=HYP_BAD,
        source_lang="en", target_lang="ja", srl=True,
    )
    assert s_good.f1 > s_bad.f1


@pytest.mark.slow
def test_readme_variant_4_ref_based_srl_meant2():
    scorer = Ryokai(sim_backend=EmbeddingSimBackend("minilm"))
    s_para = scorer.score(
        reference=REF, hypothesis=PARA,
        target_lang="ja", srl=True,
    )
    s_bad = scorer.score(
        reference=REF, hypothesis=HYP_BAD,
        target_lang="ja", srl=True,
    )
    assert s_para.f1 > s_bad.f1


@pytest.mark.slow
def test_readme_variant_5_bertscore_aligner():
    bertscore = Ryokai(
        sim_backend=EmbeddingSimBackend("minilm"),
        aligner="bertscore",
    )
    s_good = bertscore.score(
        source=SRC, hypothesis=HYP,
        source_lang="en", target_lang="ja",
    )
    s_bad = bertscore.score(
        source=SRC, hypothesis=HYP_BAD,
        source_lang="en", target_lang="ja",
    )
    assert s_good.f1 > s_bad.f1
