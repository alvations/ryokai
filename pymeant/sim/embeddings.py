"""Multilingual embedding similarity (default sentence-transformers backbone).

We use `paraphrase-multilingual-MiniLM-L12-v2` by default — it covers all
13 MEANT languages out of the box, is 110 MB, and runs on CPU.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=4)
def _load_model(model_id: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id)


class EmbeddingSimBackend:
    """Cosine similarity over multilingual sentence/phrase embeddings."""

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.model_id = model_id

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = _load_model(self.model_id)
        embs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.astype(np.float32)

    def sim(self, a: list[str], b: list[str]) -> np.ndarray:
        """Pairwise cosine similarity matrix of shape (len(a), len(b))."""
        if not a or not b:
            return np.zeros((len(a), len(b)), dtype=np.float32)
        ea = self.encode(a)
        eb = self.encode(b)
        return (ea @ eb.T).astype(np.float32)
