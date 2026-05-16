"""Predicate-argument graph: the unit MEANT scores over."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Argument:
    """One semantic argument: a labelled span filling a role for a predicate."""
    label: str           # canonical role class (e.g. "A0", "TMP")
    text: str            # the filler text
    raw_label: str = ""  # pre-aliasing label, for debugging


@dataclass
class Frame:
    """One predicate plus its arguments."""
    predicate: str                       # the verb / predicate surface form
    arguments: list[Argument] = field(default_factory=list)

    def args_by_label(self, label: str) -> list[Argument]:
        return [a for a in self.arguments if a.label == label]


@dataclass
class SRLGraph:
    """The set of semantic frames extracted from one sentence."""
    sentence: str
    frames: list[Frame] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.frames
