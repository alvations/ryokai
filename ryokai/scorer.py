"""MEANT scoring: weighted F-score over aligned predicates and arguments.

Mirrors the four-step pipeline from MEANT 2.0:

1. Parse ref + hyp into semantic frames (per-frame predicate + arguments).
2. Hungarian-match predicates by lexical similarity of the predicate text.
3. For each matched predicate pair, Hungarian-match arguments by lexical
   similarity of the filler text. Only arguments with the *same* canonical
   role class can match.
4. Weighted F-score over matched role labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import SRLGraph
from .labelconfig import LabelConfig
from .match import max_match
from .sim.embeddings import EmbeddingSimBackend


# default per-role weights. Bigger weight = bigger contribution to F.
# Heuristic: predicate (V) and core args (A0..A2) matter most; modifiers
# matter less. Unknown / OTHER carry near-zero weight.
DEFAULT_WEIGHTS: dict[str, float] = {
    "V":   1.00,
    "A0":  1.00,
    "A1":  1.00,
    "A2":  0.80,
    "TMP": 0.60,
    "LOC": 0.60,
    "PRP": 0.50,
    "MNR": 0.50,
    "EXT": 0.40,
    "MOD": 0.30,
    "NEG": 0.80,
    "OTHER": 0.10,
}


@dataclass
class Score:
    """The output of scoring one (ref, hyp) pair.

    Float-coerces to `f1` so `float(score)` returns the headline number.
    """
    precision: float
    recall: float
    f1: float
    n_frames_ref: int
    n_frames_hyp: int
    n_aligned_predicates: int

    def __float__(self) -> float:  # so float(score) returns f1
        return self.f1


# Backwards-compatible alias — historical name from the MEANT papers.
MEANTScore = Score


def _resolve_labels(graph: SRLGraph, label_config: LabelConfig) -> None:
    """Mutate the graph in-place to fill in canonical labels."""
    for frame in graph.frames:
        for arg in frame.arguments:
            if not arg.label:
                arg.label = label_config.canonical(arg.raw_label)


def _score_one(
    ref: SRLGraph,
    hyp: SRLGraph,
    sim_backend: EmbeddingSimBackend,
    weights: dict[str, float],
    label_config: LabelConfig,
) -> Score:
    _resolve_labels(ref, label_config)
    _resolve_labels(hyp, label_config)

    if not ref.frames and not hyp.frames:
        # degenerate: nothing to compare -> treat as exact match
        return Score(1.0, 1.0, 1.0, 0, 0, 0)
    if not ref.frames or not hyp.frames:
        return Score(0.0, 0.0, 0.0, len(ref.frames), len(hyp.frames), 0)

    # 1) align predicates by lexical similarity of the predicate text
    ref_preds = [f.predicate for f in ref.frames]
    hyp_preds = [f.predicate for f in hyp.frames]
    pred_sim = sim_backend.sim(ref_preds, hyp_preds)

    pred_pairs, _ = max_match(pred_sim)

    total_match = 0.0
    total_ref = 0.0
    total_hyp = 0.0

    # weight per-frame contributions by predicate similarity * V weight
    v_w = weights.get("V", 1.0)
    for i, j in pred_pairs:
        pred_score = float(pred_sim[i, j])
        if pred_score <= 0:
            continue
        ref_frame = ref.frames[i]
        hyp_frame = hyp.frames[j]

        # 2) for this frame pair: align arguments per canonical role class
        for role in {a.label for a in ref_frame.arguments} | {a.label for a in hyp_frame.arguments}:
            w = weights.get(role, weights.get("OTHER", 0.1))
            ref_args = ref_frame.args_by_label(role)
            hyp_args = hyp_frame.args_by_label(role)
            if not ref_args and not hyp_args:
                continue
            ref_texts = [a.text for a in ref_args]
            hyp_texts = [a.text for a in hyp_args]
            if ref_texts and hyp_texts:
                arg_sim = sim_backend.sim(ref_texts, hyp_texts)
                _, total = max_match(arg_sim)
            else:
                total = 0.0
            total_match += w * total
            total_ref += w * len(ref_args)
            total_hyp += w * len(hyp_args)

        # predicate itself counts as one V-matched unit
        total_match += v_w * pred_score
        total_ref += v_w
        total_hyp += v_w

    # account for unaligned ref / hyp frames (each contributes their V to denom)
    aligned_ref = {i for i, _ in pred_pairs}
    aligned_hyp = {j for _, j in pred_pairs}
    for i, frame in enumerate(ref.frames):
        if i not in aligned_ref:
            total_ref += v_w
            for a in frame.arguments:
                total_ref += weights.get(a.label, weights.get("OTHER", 0.1))
    for j, frame in enumerate(hyp.frames):
        if j not in aligned_hyp:
            total_hyp += v_w
            for a in frame.arguments:
                total_hyp += weights.get(a.label, weights.get("OTHER", 0.1))

    precision = total_match / total_hyp if total_hyp else 0.0
    recall = total_match / total_ref if total_ref else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return Score(
        precision=float(np.clip(precision, 0.0, 1.0)),
        recall=float(np.clip(recall, 0.0, 1.0)),
        f1=float(np.clip(f1, 0.0, 1.0)),
        n_frames_ref=len(ref.frames),
        n_frames_hyp=len(hyp.frames),
        n_aligned_predicates=len(pred_pairs),
    )


def score_pair(
    ref_graph: SRLGraph,
    hyp_graph: SRLGraph,
    sim_backend: EmbeddingSimBackend | None = None,
    weights: dict[str, float] | None = None,
    label_config: LabelConfig | None = None,
) -> Score:
    """Score a single (reference, hypothesis) pair already parsed into graphs."""
    return _score_one(
        ref_graph,
        hyp_graph,
        sim_backend or EmbeddingSimBackend(),
        weights or DEFAULT_WEIGHTS,
        label_config or LabelConfig(),
    )


def score_nosrl(
    reference: str,
    hypothesis: str,
    sim_backend=None,
    *,
    lang: str = "en",
    content_only: bool = False,
    aggregation: str = "f1",
    aligner: str = "sentence",
    threshold: float = 0.5,
    exact_match_shortcut: bool = True,
) -> Score:
    """No-SRL MEANT — YiSi-1 / WOLVESAAR-style.

    Bypasses the SRL graph: align reference and hypothesis tokens
    one-to-one and compute F-score (or WOLVESAAR harmonic mean) over the
    aligned-token proportions.

    Two aligners are available:

    `aligner="sentence"` (default, fast)
        Uses the sentence-level multilingual MiniLM to embed every
        whitespace token in isolation and Hungarian-match by cosine.
        Cheap, no extra model download, but token vectors are *not* in
        context — single-token "sentences" lose the surrounding cues
        that make word alignment work in Sultan et al. (2014).

    `aligner="hungarian"` | `"argmax"` | `"itermax"` (contextual)
        Uses XLM-RoBERTa contextual hidden states (`xlm-roberta-base`,
        ~1.1 GB), mean-pools subwords back to whole words, with three
        matching strategies:
            - `"hungarian"` — strict 1-to-1 (Sultan et al. style)
            - `"argmax"`    — SimAlign Inter; bidirectional argmax
                             intersection, allows many-to-many when
                             both directions agree
            - `"itermax"`   — SimAlign IterMax (recommended); argmax
                             intersection seed + iterative growth,
                             highest recall
        All three apply a Sultan-style exact-match prior (case-insensitive
        surface match boosts a pair to similarity 1.0).

    Parameters
    ----------
    reference, hypothesis : str
        The two sentences to compare.
    sim_backend : EmbeddingSimBackend | None
        Only used when `aligner="sentence"`. Defaults to multilingual MiniLM.
    lang : str
        Language for content-word filtering (only used if `content_only`).
    content_only : bool
        If True (WOLVESAAR / Sultan et al. style), drop stopwords + punctuation
        before alignment. Default False.
    aggregation : str
        `"f1"` (default) or `"harmonic"`. Both reduce to the same formula
        2·p·r/(p+r) — separate name kept so callers can be explicit about intent.
    aligner : str
        `"sentence"` (default, fast) or one of the contextual variants
        `"hungarian"` / `"argmax"` / `"itermax"`. `"itermax"` is the
        SimAlign-recommended best-quality option.
    threshold : float
        Used by `aligner="contextual"`: alignment cosine floor. Default 0.5.
    exact_match_shortcut : bool
        Used by `aligner="contextual"`: case-insensitive surface-form equality
        boosts the pair's similarity to 1.0, matching Sultan et al.'s
        lexical-prior cascade. Default True.
    """
    if aggregation not in ("f1", "harmonic"):
        raise ValueError(f"aggregation must be 'f1' or 'harmonic', got {aggregation!r}")
    contextual_methods = {"hungarian", "argmax", "itermax", "mai"}
    if aligner not in ({"sentence"} | contextual_methods):
        raise ValueError(
            f"aligner must be 'sentence' or one of {sorted(contextual_methods)}, "
            f"got {aligner!r}"
        )

    if aligner in contextual_methods:
        from .sim.contextual import ContextualTokenSimBackend
        from .stopwords import is_content_token
        ctx = ContextualTokenSimBackend()
        pairs, ref_words, hyp_words = ctx.align(
            reference,
            hypothesis,
            method=aligner,
            threshold=threshold,
            exact_match_shortcut=exact_match_shortcut,
        )
        # filter words + pairs by content-word predicate if requested
        if content_only:
            ref_keep = {i for i, w in enumerate(ref_words) if is_content_token(w, lang)}
            hyp_keep = {j for j, w in enumerate(hyp_words) if is_content_token(w, lang)}
            ref_words = [w for i, w in enumerate(ref_words) if i in ref_keep]
            hyp_words = [w for j, w in enumerate(hyp_words) if j in hyp_keep]
            pairs = [(i, j, s) for i, j, s in pairs if i in ref_keep and j in hyp_keep]
        n_ref, n_hyp = len(ref_words), len(hyp_words)
        if n_ref == 0 and n_hyp == 0:
            return Score(1.0, 1.0, 1.0, 0, 0, 0)
        if n_ref == 0 or n_hyp == 0:
            return Score(0.0, 0.0, 0.0, 0, 0, 0)
        matched_sum = sum(s for _, _, s in pairs)
        precision = matched_sum / n_hyp
        recall = matched_sum / n_ref

    else:  # aligner == "sentence"
        sim = sim_backend or EmbeddingSimBackend()
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        if content_only:
            from .stopwords import is_content_token
            ref_tokens = [t for t in ref_tokens if is_content_token(t, lang)]
            hyp_tokens = [t for t in hyp_tokens if is_content_token(t, lang)]
        if not ref_tokens and not hyp_tokens:
            return Score(1.0, 1.0, 1.0, 0, 0, 0)
        if not ref_tokens or not hyp_tokens:
            return Score(0.0, 0.0, 0.0, 0, 0, 0)
        s = sim.sim(ref_tokens, hyp_tokens)
        from .match import max_match
        _, matched_sum = max_match(s)
        precision = matched_sum / len(hyp_tokens)
        recall = matched_sum / len(ref_tokens)

    combined = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return Score(
        precision=float(np.clip(precision, 0.0, 1.0)),
        recall=float(np.clip(recall, 0.0, 1.0)),
        f1=float(np.clip(combined, 0.0, 1.0)),
        n_frames_ref=0,
        n_frames_hyp=0,
        n_aligned_predicates=0,
    )
