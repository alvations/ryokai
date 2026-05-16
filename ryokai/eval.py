"""Alignment-evaluation utilities (AER and friends).

For benchmarking ryokai's word aligners against a gold word-alignment
dataset (WPT, Hansards, Europarl etc.) using the standard
Alignment-Error-Rate metric of Och & Ney (2003).

Gold alignments come in two flavours:
    - **Sure (S)**: high-confidence one-to-one alignments.
    - **Possible (P)**: P ⊇ S; additional permissible alignments.

For a system alignment A, AER is:

    AER = 1 - ( |A ∩ S| + |A ∩ P| ) / ( |A| + |S| )

Lower is better; AER = 0 means A == S exactly.

Also provided: a small helper to sweep ryokai's aligners over a list
of (ref, hyp, gold) examples and report AER per aligner.
"""
from __future__ import annotations

from collections.abc import Iterable


Pair = tuple[int, int]
Alignment = set[Pair]


def _as_pairs(x: Iterable) -> Alignment:
    """Normalise any iterable of pairs (lists, tuples, 3-tuples) -> set[(i,j)]."""
    out: Alignment = set()
    for item in x:
        if len(item) >= 2:
            out.add((int(item[0]), int(item[1])))
    return out


def aer(
    predicted: Iterable,
    sure: Iterable,
    possible: Iterable | None = None,
) -> float:
    """Alignment Error Rate (Och & Ney 2003).

    Parameters
    ----------
    predicted : Iterable[(i, j)] or Iterable[(i, j, score)]
        Hypothesised alignments. Extra elements (e.g. similarity) are ignored.
    sure : Iterable[(i, j)]
        High-confidence gold alignments (S).
    possible : Iterable[(i, j)] | None
        Permissible alignments (P ⊇ S). If None, treated as equal to S.

    Returns
    -------
    float
        AER in [0, 1]. Returns 0.0 when both predicted and sure are empty.
    """
    A = _as_pairs(predicted)
    S = _as_pairs(sure)
    P = _as_pairs(possible) if possible is not None else set(S)
    if not P.issuperset(S):
        # tolerate it: union-in any S not in P (don't crash on bad gold)
        P = P | S
    if not A and not S:
        return 0.0
    num = len(A & S) + len(A & P)
    den = len(A) + len(S)
    if den == 0:
        return 0.0
    return 1.0 - num / den


def precision(predicted: Iterable, gold: Iterable) -> float:
    A = _as_pairs(predicted)
    G = _as_pairs(gold)
    if not A:
        return 0.0
    return len(A & G) / len(A)


def recall(predicted: Iterable, gold: Iterable) -> float:
    A = _as_pairs(predicted)
    G = _as_pairs(gold)
    if not G:
        return 0.0
    return len(A & G) / len(G)


def f1(predicted: Iterable, gold: Iterable) -> float:
    p = precision(predicted, gold)
    r = recall(predicted, gold)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_aligner(
    aligner,
    examples: Iterable[tuple[str, str, Iterable, Iterable | None]],
    *,
    method: str = "itermax",
    threshold: float = 0.5,
) -> dict:
    """Run `aligner.align(ref, hyp, ...)` over `examples` and report
    per-corpus AER / precision / recall / F1.

    `examples` is an iterable of `(ref, hyp, sure, possible)` tuples;
    `possible` may be None.

    `aligner` may be any object with the
    `align(ref, hyp, method=..., threshold=...)` signature exposed by
    `ContextualTokenSimBackend` and `StaticEmbeddingSimBackend`.

    Returns a dict with keys `n`, `aer`, `precision`, `recall`, `f1`,
    each averaged over the examples.
    """
    examples = list(examples)
    if not examples:
        return {"n": 0, "aer": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    aers, ps, rs, fs = [], [], [], []
    for ref, hyp, sure, possible in examples:
        pairs, _, _ = aligner.align(ref, hyp, method=method, threshold=threshold)
        aers.append(aer(pairs, sure, possible))
        gold = _as_pairs(sure)
        ps.append(precision(pairs, gold))
        rs.append(recall(pairs, gold))
        fs.append(f1(pairs, gold))
    return {
        "n": len(examples),
        "aer": sum(aers) / len(aers),
        "precision": sum(ps) / len(ps),
        "recall": sum(rs) / len(rs),
        "f1": sum(fs) / len(fs),
    }
