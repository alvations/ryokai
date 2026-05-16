"""ryokai — 了解.

Unified semantic machine-translation evaluation, combining the strengths
of MEANT 2.0, XMEANT, YiSi-1/2, WOLVESAAR, and SimAlign behind one clean
Python API on top of modern multilingual embeddings.

Quickstart
----------
>>> from ryokai import Ryokai
>>> r = Ryokai(lang="en")
>>> r.score(
...     reference="The cat sat on the mat.",
...     hypothesis="A cat is sitting on the mat.",
... )
Score(precision=..., recall=..., f1=..., ...)

Variants
--------
- standard MEANT (frame-based, this default)
- YiSi-1 / WOLVESAAR no-SRL mode: ``Ryokai(lang=..., use_srl=False)``
- XMEANT / YiSi-2 reference-free mode: ``r.score_xlingual(...)``
- SimAlign-style contextual word alignment: ``aligner="argmax" | "itermax" | "mai" | "hungarian"``
"""
from __future__ import annotations

from ._langs import SUPPORTED_LANGS
from .graph import Argument, Frame, SRLGraph
from .labelconfig import LabelConfig
from .scorer import DEFAULT_WEIGHTS, MEANTScore, Score, score_nosrl, score_pair
from .sim import ContextualTokenSimBackend, StaticEmbeddingSimBackend
from .sim.embeddings import EmbeddingSimBackend
from .srl import HFPOSHeuristicSRLBackend, HFTokenClassifierSRLBackend, SRLBackend

__version__ = "0.1.0"


class Ryokai:
    """Top-level MT-eval API.

    One class drives all scoring modes; what mode you get depends on the
    constructor flags and which ``score*`` method you call:

    +-------------------------------+-----------------------------------------+
    | Call                          | Mode                                    |
    +===============================+=========================================+
    | ``Ryokai(lang).score(r, h)``  | Full MEANT 2.0 — semantic frames        |
    | ``Ryokai(lang, use_srl=False) | YiSi-1 / WOLVESAAR — no SRL, word       |
    |   .score(r, h)``              | alignment + embedding F-score           |
    | ``.score_xlingual(s, h, sl)`` | XMEANT / YiSi-2 — reference-free        |
    +-------------------------------+-----------------------------------------+

    Parameters
    ----------
    lang : str
        Two-letter language code of the hypothesis (one of `SUPPORTED_LANGS`).
    use_srl : bool
        If True (default), run the full frame-based MEANT pipeline. If
        False, use the YiSi-1 / WOLVESAAR no-SRL fallback (word alignment
        + embedding F-score).
    srl_backend : SRLBackend | None
        SRL backend used when `use_srl=True`. Defaults to
        `HFPOSHeuristicSRLBackend` — a multilingual XLM-R UDPOS heuristic
        covering all 13 supported languages with one model.
    sim_backend : EmbeddingSimBackend | None
        Sentence / argument similarity backend. Defaults to multilingual
        MiniLM. Swap to any preset (``"qwen3-0.6b"``, ``"jina-v3"``,
        ``"nemotron-8b"``…) or any HF model id.
    weights : dict[str, float] | None
        Per-role weights for MEANT F-score. Defaults to `DEFAULT_WEIGHTS`.
    label_config : LabelConfig | None
        SRL label aliasing. Defaults to bundled `data/labelconfig.yaml`.
    content_only : bool
        No-SRL only: drop stopwords + punctuation before alignment
        (WOLVESAAR style).
    aggregation : str
        No-SRL only: ``"f1"`` (default) or ``"harmonic"``.
    aligner : str
        No-SRL only: ``"sentence"`` (fast, default), ``"hungarian"``,
        ``"argmax"``, ``"itermax"``, or ``"mai"``.
    threshold : float
        No-SRL contextual aligners: cosine floor below which a pair is
        discarded. Default 0.5.
    exact_match_shortcut : bool
        No-SRL contextual aligners: case-insensitive surface equality
        boosts a pair's similarity to 1.0 (Sultan-style prior).
    """

    def __init__(
        self,
        lang: str,
        use_srl: bool = True,
        srl_backend: SRLBackend | None = None,
        sim_backend: EmbeddingSimBackend | None = None,
        weights: dict[str, float] | None = None,
        label_config: LabelConfig | None = None,
        content_only: bool = False,
        aggregation: str = "f1",
        aligner: str = "sentence",
        threshold: float = 0.5,
        exact_match_shortcut: bool = True,
    ) -> None:
        if lang not in SUPPORTED_LANGS:
            raise ValueError(
                f"Language {lang!r} not supported. "
                f"Supported: {sorted(SUPPORTED_LANGS)}"
            )
        self.lang = lang
        self.use_srl = use_srl
        self.srl = srl_backend or (HFPOSHeuristicSRLBackend() if use_srl else None)
        self.sim = sim_backend or EmbeddingSimBackend()
        self.weights = weights or DEFAULT_WEIGHTS
        self.label_config = label_config or LabelConfig()
        self.content_only = content_only
        self.aggregation = aggregation
        self.aligner = aligner
        self.threshold = threshold
        self.exact_match_shortcut = exact_match_shortcut

    def parse(self, sentence: str, lang: str | None = None) -> SRLGraph:
        """Run SRL on one sentence. `lang` defaults to `self.lang`."""
        if self.srl is None:
            return SRLGraph(sentence=sentence)
        return self.srl.parse(sentence, lang or self.lang)

    def score(self, reference: str, hypothesis: str) -> Score:
        """Score a single ref/hyp MT pair. Higher = more adequate."""
        if not self.use_srl:
            return score_nosrl(
                reference,
                hypothesis,
                self.sim,
                lang=self.lang,
                content_only=self.content_only,
                aggregation=self.aggregation,
                aligner=self.aligner,
                threshold=self.threshold,
                exact_match_shortcut=self.exact_match_shortcut,
            )
        ref = self.parse(reference)
        hyp = self.parse(hypothesis)
        return score_pair(ref, hyp, self.sim, self.weights, self.label_config)

    def score_corpus(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> list[Score]:
        """Score a parallel corpus — one Score per (ref, hyp) pair."""
        if len(references) != len(hypotheses):
            raise ValueError(
                f"len(references)={len(references)} != len(hypotheses)={len(hypotheses)}"
            )
        return [self.score(r, h) for r, h in zip(references, hypotheses)]

    def score_xlingual(
        self,
        source: str,
        hypothesis: str,
        source_lang: str,
    ) -> Score:
        """XMEANT / YiSi-2 style **reference-free** scoring.

        Scores `hypothesis` (in `self.lang`) directly against `source`
        (in `source_lang`) using cross-lingual semantic-frame agreement.
        No reference translation needed — works because both SRL and
        similarity backends are multilingual.

        Cite: Lo et al. 2014 (XMEANT, P14-2124) and Lo 2019 (YiSi, WMT19).
        """
        if source_lang not in SUPPORTED_LANGS:
            raise ValueError(
                f"source_lang {source_lang!r} not supported. "
                f"Supported: {sorted(SUPPORTED_LANGS)}"
            )
        if not self.use_srl:
            return score_nosrl(
                source,
                hypothesis,
                self.sim,
                lang=source_lang,
                content_only=self.content_only,
                aggregation=self.aggregation,
                aligner=self.aligner,
                threshold=self.threshold,
                exact_match_shortcut=self.exact_match_shortcut,
            )
        src_graph = self.parse(source, lang=source_lang)
        hyp_graph = self.parse(hypothesis)
        return score_pair(
            src_graph,
            hyp_graph,
            self.sim,
            self.weights,
            self.label_config,
        )


# Backwards-compatible alias — most MT-eval literature calls this MEANT.
MEANT = Ryokai

__all__ = [
    "Ryokai",
    "MEANT",
    "Score",
    "MEANTScore",
    "SRLGraph",
    "Frame",
    "Argument",
    "LabelConfig",
    "SRLBackend",
    "HFPOSHeuristicSRLBackend",
    "HFTokenClassifierSRLBackend",
    "EmbeddingSimBackend",
    "ContextualTokenSimBackend",
    "StaticEmbeddingSimBackend",
    "DEFAULT_WEIGHTS",
    "SUPPORTED_LANGS",
    "score_pair",
    "score_nosrl",
]
