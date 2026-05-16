"""pymeant — Python port of MEANT 2.0.

Quickstart
----------
>>> from pymeant import MEANT
>>> meant = MEANT(lang="en")
>>> meant.score(
...     reference="The cat sat on the mat.",
...     hypothesis="A cat is sitting on the mat.",
... )
MEANTScore(precision=..., recall=..., f1=..., ...)

Variants supported (see README):
- standard MEANT (this default)
- YiSi-1 style no-SRL mode: `MEANT(lang=..., use_srl=False)`
- XMEANT / YiSi-2 style reference-free mode: `meant.score_xlingual(...)`
"""
from __future__ import annotations

from ._langs import SUPPORTED_LANGS
from .graph import Argument, Frame, SRLGraph
from .labelconfig import LabelConfig
from .scorer import DEFAULT_WEIGHTS, MEANTScore, score_nosrl, score_pair
from .sim.embeddings import EmbeddingSimBackend
from .srl import HFPOSHeuristicSRLBackend, HFTokenClassifierSRLBackend, SRLBackend

__version__ = "0.1.0"

__all__ = [
    "MEANT",
    "MEANTScore",
    "SRLGraph",
    "Frame",
    "Argument",
    "LabelConfig",
    "SRLBackend",
    "HFPOSHeuristicSRLBackend",
    "HFTokenClassifierSRLBackend",
    "EmbeddingSimBackend",
    "DEFAULT_WEIGHTS",
    "SUPPORTED_LANGS",
    "score_pair",
    "score_nosrl",
]


class MEANT:
    """Top-level MT-eval API.

    Parameters
    ----------
    lang : str
        Two-letter language code (one of `SUPPORTED_LANGS`).
    use_srl : bool
        If True (default), use the SRL backend and run full MEANT
        frame-based scoring. If False, fall back to YiSi-1-style
        embedding-only token alignment — no SRL needed, faster, fewer
        downloads.
    srl_backend : SRLBackend | None
        SRL backend to use when `use_srl=True`. Defaults to
        `HFPOSHeuristicSRLBackend` — pure HuggingFace transformers
        (multilingual XLM-R POS), covers all 13 MEANT languages with
        one model. Plug in `HFTokenClassifierSRLBackend(model_id)` for
        a proper PropBank-grade SRL model.
    sim_backend : EmbeddingSimBackend | None
        Lexical similarity backend. Defaults to multilingual
        sentence-transformers MiniLM (also covers all 13 langs in one
        110 MB model and supports cross-lingual sim out of the box).
    weights : dict[str, float] | None
        Per-role weights. Defaults to `DEFAULT_WEIGHTS`.
    label_config : LabelConfig | None
        SRL label aliasing. Defaults to the bundled `data/labelconfig.yaml`.
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

    def score(self, reference: str, hypothesis: str) -> MEANTScore:
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
    ) -> list[MEANTScore]:
        """Score a parallel corpus — one MEANTScore per pair."""
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
    ) -> MEANTScore:
        """XMEANT / YiSi-2 style **reference-free** scoring.

        Scores `hypothesis` (in `self.lang`) directly against `source`
        (in `source_lang`) using cross-lingual semantic-frame agreement.
        No reference translation needed — works because both SRL backend
        and similarity backend handle multiple languages in one model.

        Cite: Lo et al. 2014 "XMEANT" (P14-2124) and Lo 2019 "YiSi" (WMT19).
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
