"""Static word-embedding backend.

Pure-lookup word vectors — no transformer forward pass — for the
SimAlign "static embedding" lane. Two sources are supported:

1. **HF transformer embedding layer** (default): take the input
   `word_embeddings` of any HF model on the Hub. This gives a static
   token vector per BPE/word-piece, mean-pooled back to whole words.
   Same multilingual coverage as XLM-R / mBERT, no GPU needed, much
   faster than the contextual forward pass.

2. **fastText / gensim KeyedVectors** (optional): if `gensim` is
   installed and a `.vec` / `.bin` path is given, load fastText vectors
   directly. This matches SimAlign's original "static" backend.

Both produce the same `align(ref, hyp, ...)` API as
`ContextualTokenSimBackend`, so they're drop-in compatible everywhere
the contextual aligners are used.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

from .contextual import (  # reuse Sultan/SimAlign matching logic
    _itermax,
)

DEFAULT_STATIC_MODEL = "xlm-roberta-base"


@lru_cache(maxsize=4)
def _load_hf_embed_layer(model_id: str):
    """Return (tokenizer, embedding_matrix np.ndarray) for an HF model."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    # Walk to the input embedding matrix — works for BERT/XLM-R/Qwen/etc.
    emb = mdl.get_input_embeddings()
    matrix = emb.weight.detach().cpu().numpy().astype(np.float32)
    # L2 normalise per-row so dot product == cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tok, matrix / norms


@lru_cache(maxsize=2)
def _load_fasttext(path: str):
    """Load a fastText / word2vec vector file via gensim, if installed."""
    try:
        from gensim.models import KeyedVectors
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Loading fastText vectors requires gensim: pip install gensim"
        ) from e
    kv = KeyedVectors.load_word2vec_format(
        path, binary=path.endswith(".bin"), no_header=path.endswith(".vec"),
    )
    return kv


def _embed_words_hf_static(sentence: str, model_id: str):
    tok, mat = _load_hf_embed_layer(model_id)
    enc = tok(sentence, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    ids = enc["input_ids"]
    words: list[str] = []
    vecs: list[np.ndarray] = []
    cur_chars: list[str] = []
    cur_pieces: list[np.ndarray] = []
    prev_end = -1
    for i, (start, end) in enumerate(offsets):
        if start == end == 0:
            continue
        v = mat[ids[i]]
        if start > prev_end:
            if cur_pieces:
                words.append("".join(cur_chars))
                vecs.append(np.mean(cur_pieces, axis=0))
            cur_chars = [sentence[start:end]]
            cur_pieces = [v]
        else:
            cur_chars.append(sentence[start:end])
            cur_pieces.append(v)
        prev_end = end
    if cur_pieces:
        words.append("".join(cur_chars))
        vecs.append(np.mean(cur_pieces, axis=0))
    if not vecs:
        return [], np.zeros((0, 0), dtype=np.float32)
    out = np.vstack(vecs).astype(np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return words, out / norms


def _embed_words_fasttext(sentence: str, kv) -> tuple[list[str], np.ndarray]:
    words: list[str] = []
    vecs: list[np.ndarray] = []
    for w in sentence.split():
        if w in kv:
            v = kv[w]
        elif w.lower() in kv:
            v = kv[w.lower()]
        else:
            continue  # OOV
        words.append(w)
        vecs.append(v)
    if not vecs:
        return [], np.zeros((0, 0), dtype=np.float32)
    out = np.vstack(vecs).astype(np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return words, out / norms


class StaticEmbeddingSimBackend:
    """Static word vectors + same alignment methods as the contextual backend.

    Parameters
    ----------
    model_id : str
        HF model id (or `CTX_PRESETS` shorthand); the model's input
        embedding layer is used. Default: `xlm-roberta-base`. Pass
        e.g. `"qwen3-0.6b"` to use Qwen3's input embeddings instead.
    fasttext_path : str | None
        Optional path to a fastText `.vec` / `.bin` file; if set, this
        is used instead of the HF embedding layer (requires `gensim`).
    max_distortion : float | None
        SimAlign-style positional distortion filter (see
        `ContextualTokenSimBackend`).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_STATIC_MODEL,
        fasttext_path: str | None = None,
        max_distortion: float | None = None,
    ) -> None:
        from .contextual import CTX_PRESETS
        self.model_id = CTX_PRESETS.get(model_id, model_id)
        self.fasttext_path = fasttext_path
        self.max_distortion = max_distortion

    def _embed(self, sentence: str):
        if self.fasttext_path:
            kv = _load_fasttext(self.fasttext_path)
            return _embed_words_fasttext(sentence, kv)
        return _embed_words_hf_static(sentence, self.model_id)

    def align(
        self,
        ref: str,
        hyp: str,
        method: str = "itermax",
        threshold: float = 0.5,
        exact_match_shortcut: bool = True,
    ) -> tuple[list[tuple[int, int, float]], list[str], list[str]]:
        if method not in ("hungarian", "argmax", "itermax", "mai"):
            raise ValueError(f"unknown align method {method!r}")

        ref_words, ref_vecs = self._embed(ref)
        hyp_words, hyp_vecs = self._embed(hyp)
        if not ref_words or not hyp_words:
            return [], ref_words, hyp_words

        sim = (ref_vecs @ hyp_vecs.T).astype(np.float32)

        if exact_match_shortcut:
            ref_low = [w.lower() for w in ref_words]
            hyp_low = [w.lower() for w in hyp_words]
            for i, rw in enumerate(ref_low):
                for j, hw in enumerate(hyp_low):
                    if rw == hw and rw:
                        sim[i, j] = 1.0

        pairs = _dispatch_match(sim, method, threshold)

        if self.max_distortion is not None and pairs:
            n_ref, n_hyp = len(ref_words), len(hyp_words)
            pairs = [
                (i, j, s) for i, j, s in pairs
                if abs(i / max(n_ref - 1, 1) - j / max(n_hyp - 1, 1)) <= self.max_distortion
            ]
        return pairs, ref_words, hyp_words


def _dispatch_match(sim: np.ndarray, method: str, threshold: float):
    """Shared matcher dispatch used by both static + contextual backends."""
    from scipy.optimize import linear_sum_assignment

    if method == "hungarian":
        rows, cols = linear_sum_assignment(-sim)
        return [
            (int(i), int(j), float(sim[i, j]))
            for i, j in zip(rows.tolist(), cols.tolist())
            if sim[i, j] >= threshold
        ]
    if method == "argmax":
        fwd = sim.argmax(axis=1)
        rev = sim.argmax(axis=0)
        return [
            (i, int(j), float(sim[i, j]))
            for i, j in enumerate(fwd.tolist())
            if rev[j] == i and sim[i, j] >= threshold
        ]
    if method == "itermax":
        return _itermax(sim, threshold=threshold)
    if method == "mai":
        # SimAlign "mai" = union of match (hungarian) + argmax + itermax
        seen: set[tuple[int, int]] = set()
        out: list[tuple[int, int, float]] = []
        for m in ("hungarian", "argmax", "itermax"):
            for i, j, s in _dispatch_match(sim, m, threshold):
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                out.append((i, j, s))
        return sorted(out)
    raise ValueError(method)
