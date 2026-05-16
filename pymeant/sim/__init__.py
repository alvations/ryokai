"""Similarity backends — multilingual sentence embedding cosine by default."""
from __future__ import annotations

from .contextual import ContextualTokenSimBackend
from .embeddings import EmbeddingSimBackend

__all__ = ["EmbeddingSimBackend", "ContextualTokenSimBackend"]
