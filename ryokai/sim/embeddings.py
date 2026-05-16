"""Multilingual embedding similarity for predicate / argument scoring.

Default backbone is `paraphrase-multilingual-MiniLM-L12-v2` — 110 MB,
runs on CPU, covers all 13 MEANT languages. For higher quality, swap in
a recent multilingual embedding model via the `model_id` argument or one
of the shorthand presets in `EMBEDDING_PRESETS`.

Recommended modern alternatives (all loadable through `sentence-transformers`
or, if it doesn't recognise them, via `transformers` AutoModel mean-pooling):

| Preset          | HF model id                                        | Size  |
| --------------- | -------------------------------------------------- | ----- |
| `minilm`        | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 110 MB |
| `mpnet`         | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 1.1 GB |
| `qwen3-0.6b`    | Qwen/Qwen3-Embedding-0.6B                          | 1.2 GB |
| `qwen3-4b`      | Qwen/Qwen3-Embedding-4B                            | 8 GB   |
| `jina-v3`       | jinaai/jina-embeddings-v3                          | 570 MB |
| `jina-v2-base`  | jinaai/jina-embeddings-v2-base-en                  | 170 MB |
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

EMBEDDING_PRESETS: dict[str, str] = {
    # 2022-2024 baselines (small + reliable)
    "minilm":             "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet":              "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "me5-large":          "intfloat/multilingual-e5-large",
    "me5-large-instruct": "intfloat/multilingual-e5-large-instruct",
    "bge-m3":             "BAAI/bge-m3",
    "mxbai-large":        "mixedbread-ai/mxbai-embed-large-v1",
    # 2024-2025 releases
    "snowflake-arctic-v2": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "jina-v2-base":       "jinaai/jina-embeddings-v2-base-en",
    "jina-v3":            "jinaai/jina-embeddings-v3",
    "nomic-v2":           "nomic-ai/nomic-embed-text-v2-moe",
    # 2025 SOTA — Qwen3 line (multilingual, top of MTEB June 2025)
    "qwen3-0.6b":         "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-4b":           "Qwen/Qwen3-Embedding-4B",
    "qwen3-8b":           "Qwen/Qwen3-Embedding-8B",
    # 2025 SOTA — NVIDIA Llama-Embed-Nemotron (Oct 2025, #1 MMTEB)
    "nemotron-8b":        "nvidia/llama-embed-nemotron-8b",
    # 2025 — Google EmbeddingGemma (text-only, multilingual)
    "embedding-gemma":    "google/embeddinggemma-300m",
}

DEFAULT_MODEL = EMBEDDING_PRESETS["minilm"]


@lru_cache(maxsize=4)
def _load_model(model_id: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, trust_remote_code=True)


class EmbeddingSimBackend:
    """Cosine similarity over multilingual sentence/phrase embeddings.

    Pass any sentence-transformers-compatible HF model id, or one of the
    shorthand keys in `EMBEDDING_PRESETS`:

        EmbeddingSimBackend("qwen3-0.6b")
        EmbeddingSimBackend("jina-v3")
        EmbeddingSimBackend("BAAI/bge-m3")
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.model_id = EMBEDDING_PRESETS.get(model_id, model_id)

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
