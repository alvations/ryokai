"""SRL backends — produce SRLGraph from a sentence."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..graph import SRLGraph


class SRLBackend(ABC):
    """Abstract base class for an SRL backend."""

    @abstractmethod
    def parse(self, sentence: str, lang: str) -> SRLGraph:
        """Return the semantic frames of one sentence."""


from .hf_pos import HFPOSHeuristicSRLBackend  # noqa: E402
from .hf_backend import HFTokenClassifierSRLBackend  # noqa: E402

__all__ = [
    "SRLBackend",
    "HFPOSHeuristicSRLBackend",
    "HFTokenClassifierSRLBackend",
]
