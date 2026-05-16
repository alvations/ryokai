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
class MEANTScore:
    """The output of MEANT scoring for one (ref, hyp) pair."""
    precision: float
    recall: float
    f1: float
    n_frames_ref: int
    n_frames_hyp: int
    n_aligned_predicates: int

    def __float__(self) -> float:  # so float(score) returns f1
        return self.f1


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
) -> MEANTScore:
    _resolve_labels(ref, label_config)
    _resolve_labels(hyp, label_config)

    if not ref.frames and not hyp.frames:
        # degenerate: nothing to compare -> treat as exact match
        return MEANTScore(1.0, 1.0, 1.0, 0, 0, 0)
    if not ref.frames or not hyp.frames:
        return MEANTScore(0.0, 0.0, 0.0, len(ref.frames), len(hyp.frames), 0)

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
    return MEANTScore(
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
) -> MEANTScore:
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
    sim_backend: EmbeddingSimBackend | None = None,
) -> MEANTScore:
    """No-SRL MEANT — YiSi-1 style.

    Bypasses the SRL graph entirely: token-level Hungarian matching of
    reference vs hypothesis words by embedding cosine, then F-score.
    Useful when SRL is unreliable or unavailable for the target language.
    """
    sim = sim_backend or EmbeddingSimBackend()
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens and not hyp_tokens:
        return MEANTScore(1.0, 1.0, 1.0, 0, 0, 0)
    if not ref_tokens or not hyp_tokens:
        return MEANTScore(0.0, 0.0, 0.0, 0, 0, 0)
    s = sim.sim(ref_tokens, hyp_tokens)
    from .match import max_match
    _, matched = max_match(s)
    precision = matched / len(hyp_tokens)
    recall = matched / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return MEANTScore(
        precision=float(np.clip(precision, 0.0, 1.0)),
        recall=float(np.clip(recall, 0.0, 1.0)),
        f1=float(np.clip(f1, 0.0, 1.0)),
        n_frames_ref=0,
        n_frames_hyp=0,
        n_aligned_predicates=0,
    )
