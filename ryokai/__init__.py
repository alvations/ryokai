"""ryokai — 了解.

Unified semantic machine-translation evaluation. Reference-free or
reference-based, word-alignment + embedding by default, frame-based
MEANT 2.0 on opt-in — all behind a single `.score()` call.

Quickstart
----------
>>> from ryokai import Ryokai
>>> scorer = Ryokai()
>>>
>>> # Reference-free (most common; XMEANT / YiSi-2 / Doc-embedding adequacy)
>>> scorer.score(source=src, hypothesis=hyp,
...              source_lang="en", target_lang="ja")
>>>
>>> # Reference-based, word alignment + embeddings
>>> # (Doc-embedding adequacy / WOLVESAAR / YiSi-1 / SimAlign)
>>> scorer.score(reference=ref, hypothesis=hyp, target_lang="ja")
>>>
>>> # Frame-based MEANT 2.0 — opt in with srl=True
>>> scorer.score(reference=ref, hypothesis=hyp, target_lang="ja", srl=True)
>>>
>>> # Frame-based reference-free (XMEANT proper)
>>> scorer.score(source=src, hypothesis=hyp,
...              source_lang="en", target_lang="ja", srl=True)
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
    """Unified MT-eval scorer.

    Construct once with the backend / alignment configuration you want;
    call ``.score(...)`` per-pair. The mode is chosen by which arguments
    you pass:

    +------------------------+--------------+--------------------------------------------+
    | source / reference     | srl          | Mode                                       |
    +========================+==============+============================================+
    | ``source=``            | False (def.) | Reference-free, word alignment + embedding |
    |                        |              | (XMEANT-lite / YiSi-2 / Doc-embedding adq) |
    +------------------------+--------------+--------------------------------------------+
    | ``reference=``         | False (def.) | Reference-based, word alignment + embedding|
    |                        |              | (WOLVESAAR / YiSi-1 / SimAlign style)      |
    +------------------------+--------------+--------------------------------------------+
    | ``source=``            | True         | Reference-free, frame-based (XMEANT)       |
    +------------------------+--------------+--------------------------------------------+
    | ``reference=``         | True         | Reference-based, frame-based (MEANT 2.0)   |
    +------------------------+--------------+--------------------------------------------+

    Constructor parameters configure the *how* (which models, which
    aligner, weights, etc.); per-call parameters configure the *what*
    (which sentences, which languages, ref- vs source-based, with or
    without SRL).

    Parameters
    ----------
    srl_backend : SRLBackend | None
        SRL backend used whenever a `.score(..., srl=True)` call is made.
        Defaults to `HFPOSHeuristicSRLBackend` — a multilingual XLM-R
        UDPOS heuristic covering all 13 supported languages with one
        model.
    sim_backend : EmbeddingSimBackend | None
        Sentence / argument similarity backend. Defaults to multilingual
        MiniLM. Swap to any preset (``"qwen3-0.6b"``, ``"jina-v3"``,
        ``"nemotron-8b"``…) or any HF model id.
    weights : dict[str, float] | None
        Per-role weights for frame-based F-score. Defaults to
        `DEFAULT_WEIGHTS`.
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
        self._srl_backend = srl_backend
        self.sim = sim_backend or EmbeddingSimBackend()
        self.weights = weights or DEFAULT_WEIGHTS
        self.label_config = label_config or LabelConfig()
        self.content_only = content_only
        self.aggregation = aggregation
        self.aligner = aligner
        self.threshold = threshold
        self.exact_match_shortcut = exact_match_shortcut

    @property
    def srl(self) -> SRLBackend:
        """Lazily instantiate the default SRL backend so users who never
        call with ``srl=True`` never pay for the SRL model download."""
        if self._srl_backend is None:
            self._srl_backend = HFPOSHeuristicSRLBackend()
        return self._srl_backend

    @staticmethod
    def _check_lang(name: str, value: str | None) -> None:
        if value is None:
            return
        if value not in SUPPORTED_LANGS:
            raise ValueError(
                f"{name}={value!r} not supported. "
                f"Supported: {sorted(SUPPORTED_LANGS)}"
            )

    def score(
        self,
        *,
        hypothesis: str,
        target_lang: str,
        source: str | None = None,
        source_lang: str | None = None,
        reference: str | None = None,
        srl: bool = False,
    ) -> Score:
        """Score one MT pair. See class docstring for the dispatch table.

        Exactly one of ``source=`` or ``reference=`` must be given.
        Higher returned ``f1`` = more adequate.
        """
        if (source is None) == (reference is None):
            raise ValueError(
                "pass exactly one of source= (reference-free) or "
                "reference= (reference-based)"
            )
        self._check_lang("target_lang", target_lang)
        self._check_lang("source_lang", source_lang)

        if srl:
            return self._score_with_srl(
                hypothesis=hypothesis,
                target_lang=target_lang,
                source=source,
                source_lang=source_lang,
                reference=reference,
            )
        return self._score_nosrl(
            hypothesis=hypothesis,
            target_lang=target_lang,
            source=source,
            source_lang=source_lang,
            reference=reference,
        )

    def _score_nosrl(
        self,
        *,
        hypothesis: str,
        target_lang: str,
        source: str | None,
        source_lang: str | None,
        reference: str | None,
    ) -> Score:
        # the "other side" of the comparison is either source or reference
        other = source if source is not None else reference
        other_lang = source_lang if source is not None else target_lang
        return score_nosrl(
            other,
            hypothesis,
            self.sim,
            lang=other_lang or target_lang,
            content_only=self.content_only,
            aggregation=self.aggregation,
            aligner=self.aligner,
            threshold=self.threshold,
            exact_match_shortcut=self.exact_match_shortcut,
        )

    def _score_with_srl(
        self,
        *,
        hypothesis: str,
        target_lang: str,
        source: str | None,
        source_lang: str | None,
        reference: str | None,
    ) -> Score:
        hyp_graph = self.srl.parse(hypothesis, target_lang)
        if source is not None:
            ref_graph = self.srl.parse(source, source_lang or target_lang)
        else:
            ref_graph = self.srl.parse(reference, target_lang)
        return score_pair(
            ref_graph, hyp_graph, self.sim, self.weights, self.label_config,
        )

    def score_corpus(
        self,
        *,
        hypotheses: list[str],
        target_lang: str,
        sources: list[str] | None = None,
        source_lang: str | None = None,
        references: list[str] | None = None,
        srl: bool = False,
    ) -> list[Score]:
        """Score a parallel corpus — one Score per pair.

        Same dispatch rules as `.score()`: pass either `sources=` or
        `references=`, not both.
        """
        if (sources is None) == (references is None):
            raise ValueError(
                "pass exactly one of sources= or references="
            )
        n = len(hypotheses)
        others = sources if sources is not None else references
        if len(others) != n:
            raise ValueError(
                f"length mismatch: hypotheses={n}, "
                f"{'sources' if sources is not None else 'references'}={len(others)}"
            )
        out: list[Score] = []
        for i in range(n):
            kwargs = dict(
                hypothesis=hypotheses[i],
                target_lang=target_lang,
                source_lang=source_lang,
                srl=srl,
            )
            if sources is not None:
                kwargs["source"] = sources[i]
            else:
                kwargs["reference"] = references[i]
            out.append(self.score(**kwargs))
        return out


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
