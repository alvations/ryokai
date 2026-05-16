"""Contextual word-level similarity, à la Sultan et al. (2014) monolingual aligner.

`EmbeddingSimBackend` (sentence-level pooling) is fine for predicate /
argument span similarity but loses word-in-context information when used
on single tokens. For YiSi-1 / WOLVESAAR style word alignment we want
**contextual** token vectors — what XLM-RoBERTa was actually trained to
produce — and a similarity floor.

This backend gives you that:

    ContextualTokenSimBackend("xlm-roberta-base")
        .tokenize_aligned(sentence)  -> [(word, vec), ...]
        .align(ref_sentence, hyp_sentence,
               threshold=0.5,
               exact_match_shortcut=True)
        -> ([(ref_idx, hyp_idx, sim), ...],   # aligned pairs
            [ref_word, ...],                  # ref token list
            [hyp_word, ...])                  # hyp token list

It is consumed by `score_nosrl(..., aligner="contextual")` for properly
monolingual-aligned WOLVESAAR scoring.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_CTX_MODEL = "xlm-roberta-base"

CTX_PRESETS: dict[str, str] = {
    # 2018-2020 baselines (proven for word alignment per the SimAlign paper)
    "mbert":        "bert-base-multilingual-cased",
    "xlm-r-base":   "xlm-roberta-base",
    "xlm-r-large":  "xlm-roberta-large",
    "mdeberta-v3":  "microsoft/mdeberta-v3-base",
    # 2024-2025 contextual encoders
    "bge-m3":             "BAAI/bge-m3",
    "snowflake-arctic-v2": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "jina-v2-base":       "jinaai/jina-embeddings-v2-base-en",
    "jina-v3":            "jinaai/jina-embeddings-v3",
    # 2025 — Qwen3 (decoder, last hidden state per token)
    "qwen3-0.6b":   "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-4b":     "Qwen/Qwen3-Embedding-4B",
    "qwen3-8b":     "Qwen/Qwen3-Embedding-8B",
    # 2025 — NVIDIA Llama-Embed-Nemotron-8B (#1 MMTEB Oct 2025)
    "nemotron-8b":  "nvidia/llama-embed-nemotron-8b",
    "embedding-gemma": "google/embeddinggemma-300m",
}


@lru_cache(maxsize=4)
def _load(model_id: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_id, trust_remote_code=True, output_hidden_states=True)
    mdl.eval()
    return tok, mdl, torch


def _embed_words(sentence: str, model_id: str, layer: int = -1):
    """Encode `sentence` with the chosen transformer, mean-pool subwords
    back to whole words. `layer` selects which hidden-state layer to use
    (-1 = last; SimAlign found middle layers, e.g. 8 for mBERT-base, work
    better for word alignment).

    Returns (words: list[str], vecs: np.ndarray shape (n_words, hidden))."""
    tok, mdl, torch = _load(model_id)
    enc = tok(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        outputs = mdl(**enc)
    if layer == -1:
        out = outputs.last_hidden_state[0]
    else:
        # hidden_states is a tuple of (n_layers+1) tensors, embedding + each layer
        hs = getattr(outputs, "hidden_states", None)
        if hs is None:
            raise RuntimeError(
                f"model {model_id!r} did not return hidden_states; "
                "cannot select a non-final layer"
            )
        out = hs[layer][0]   # (seq_len, hidden)

    # group subword tokens that share a word boundary in the original sentence
    words: list[str] = []
    vecs: list[np.ndarray] = []
    cur_word_chars: list[str] = []
    cur_pieces: list = []
    prev_end = -1
    for i, (start, end) in enumerate(offsets):
        if start == end == 0:          # special token
            continue
        if start > prev_end:           # new word boundary (preceded by space)
            if cur_pieces:
                words.append("".join(cur_word_chars))
                vecs.append(torch.stack(cur_pieces).mean(0).cpu().numpy())
            cur_word_chars = [sentence[start:end]]
            cur_pieces = [out[i]]
        else:                          # continuation of current word
            cur_word_chars.append(sentence[start:end])
            cur_pieces.append(out[i])
        prev_end = end
    if cur_pieces:
        words.append("".join(cur_word_chars))
        vecs.append(torch.stack(cur_pieces).mean(0).cpu().numpy())

    if not vecs:
        return [], np.zeros((0, 0), dtype=np.float32)
    mat = np.vstack(vecs).astype(np.float32)
    # L2 normalise so dot product is cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return words, mat / norms


class ContextualTokenSimBackend:
    """XLM-R contextual word vectors + cosine alignment with threshold.

    Closer to Sultan et al. (2014) "Back to Basics for Monolingual Alignment":
        - exact / case-insensitive equality short-circuits as in
          the Sultan et al. lexical priors (PPDB, stems);
        - cosine over contextual vectors covers paraphrase / synonymy
          (the modern equivalent of PPDB + WordNet rules);
        - Hungarian assignment enforces 1-to-1 alignment;
        - threshold below which a pair is *not* considered aligned.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_CTX_MODEL,
        layer: int = -1,
        max_distortion: int | None = None,
    ) -> None:
        """`model_id` is an HF repo id or a key from `CTX_PRESETS`.

        `layer` picks which hidden layer to use for token vectors (-1 = last).
        SimAlign found middle layers (e.g. 8 for mBERT-base, 8 for XLM-R-base)
        give better word alignment than the last layer.

        `max_distortion` drops aligned pairs whose 0..1-normalised position
        differs by more than this fraction (e.g. 0.4 drops alignments where
        word i in ref is far from its aligned hyp word). None = no filter."""
        self.model_id = CTX_PRESETS.get(model_id, model_id)
        self.layer = layer
        self.max_distortion = max_distortion

    def align(
        self,
        ref: str,
        hyp: str,
        method: str = "itermax",
        threshold: float = 0.5,
        exact_match_shortcut: bool = True,
    ) -> tuple[list[tuple[int, int, float]], list[str], list[str]]:
        """Align ref/hyp words using a contextual encoder + chosen matcher.

        Methods (drawn from SimAlign + Sultan et al.):
            "hungarian"  — scipy max-weight bipartite (1-to-1). Sultan-style.
            "argmax"     — each src word picks best tgt; intersect with reverse
                            direction (SimAlign "ArgMax" / Inter). Allows
                            many-to-one when the bidirection agrees on a tgt.
            "itermax"    — start from argmax intersection, then iteratively
                            grow alignments to recover unaligned words
                            (SimAlign "IterMax"). Best F1 in the paper.

        Returns (pairs, ref_words, hyp_words) where pairs is
        [(ref_idx, hyp_idx, similarity), ...] with similarity >= threshold.
        """
        if method not in ("hungarian", "argmax", "itermax"):
            raise ValueError(f"unknown align method {method!r}")

        ref_words, ref_vecs = _embed_words(ref, self.model_id, self.layer)
        hyp_words, hyp_vecs = _embed_words(hyp, self.model_id, self.layer)
        if not ref_words or not hyp_words:
            return [], ref_words, hyp_words

        sim = (ref_vecs @ hyp_vecs.T).astype(np.float32)

        # Sultan-style prior: case-insensitive surface-form equality -> sim = 1
        if exact_match_shortcut:
            ref_low = [w.lower() for w in ref_words]
            hyp_low = [w.lower() for w in hyp_words]
            for i, rw in enumerate(ref_low):
                for j, hw in enumerate(hyp_low):
                    if rw == hw and rw:
                        sim[i, j] = 1.0

        if method == "hungarian":
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(-sim)
            pairs = [
                (int(i), int(j), float(sim[i, j]))
                for i, j in zip(rows.tolist(), cols.tolist())
                if sim[i, j] >= threshold
            ]
        elif method == "argmax":
            # SimAlign Inter (ArgMax): pair (i,j) iff j = argmax_j' sim[i,j']
            # AND i = argmax_i' sim[i',j]. Allows many-to-many only when both
            # directions vote it through.
            fwd = sim.argmax(axis=1)        # for each ref i, best hyp
            rev = sim.argmax(axis=0)        # for each hyp j, best ref
            pairs = []
            for i, j in enumerate(fwd.tolist()):
                if rev[j] == i and sim[i, j] >= threshold:
                    pairs.append((i, int(j), float(sim[i, j])))
        else:  # itermax
            pairs = _itermax(sim, threshold=threshold)

        # SimAlign-style distortion filter: optionally drop alignments that
        # are very far apart in normalised position.
        if self.max_distortion is not None and pairs:
            n_ref, n_hyp = len(ref_words), len(hyp_words)
            pairs = [
                (i, j, s)
                for i, j, s in pairs
                if abs(i / max(n_ref - 1, 1) - j / max(n_hyp - 1, 1)) <= self.max_distortion
            ]
        return pairs, ref_words, hyp_words


def _itermax(sim: np.ndarray, threshold: float, n_iter: int = 2) -> list[tuple[int, int, float]]:
    """SimAlign IterMax: ArgMax intersection seed, then iteratively grow
    alignments to cover words still unaligned in either direction."""
    fwd = sim.argmax(axis=1)
    rev = sim.argmax(axis=0)

    aligned: set[tuple[int, int]] = set()
    for i, j in enumerate(fwd.tolist()):
        if rev[j] == i and sim[i, j] >= threshold:
            aligned.add((i, int(j)))

    n_ref, n_hyp = sim.shape
    for _ in range(n_iter):
        used_i = {i for i, _ in aligned}
        used_j = {j for _, j in aligned}
        # for each unaligned ref word, pick best still-unaligned hyp
        for i in range(n_ref):
            if i in used_i:
                continue
            cand = [(j, sim[i, j]) for j in range(n_hyp) if j not in used_j]
            if not cand:
                continue
            j_best, s_best = max(cand, key=lambda t: t[1])
            if s_best >= threshold:
                aligned.add((i, j_best))
                used_j.add(j_best)
        # symmetric: for each unaligned hyp, pick best still-unaligned ref
        used_i = {i for i, _ in aligned}
        used_j = {j for _, j in aligned}
        for j in range(n_hyp):
            if j in used_j:
                continue
            cand = [(i, sim[i, j]) for i in range(n_ref) if i not in used_i]
            if not cand:
                continue
            i_best, s_best = max(cand, key=lambda t: t[1])
            if s_best >= threshold:
                aligned.add((i_best, j))
                used_i.add(i_best)

    return [(i, j, float(sim[i, j])) for i, j in sorted(aligned)]
