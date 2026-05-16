"""Similarity backends — multilingual sentence embedding cosine by default."""
from __future__ import annotations

from .contextual import ContextualTokenSimBackend
from .embeddings import EmbeddingSimBackend
from .static import StaticEmbeddingSimBackend

__all__ = [
    "EmbeddingSimBackend",
    "ContextualTokenSimBackend",
    "StaticEmbeddingSimBackend",
]
