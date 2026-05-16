"""Load the SRL role-label aliasing table from data/labelconfig.yaml.

The labelconfig collapses SRL labels from multiple frameworks (PropBank,
Chinese PropBank, AnCora, reference/continuation tags) into one canonical
class so that MEANT can be computed across languages with a single
scoring config — same trick as the original MEANT 2.0 `meant.labelconfig`.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import yaml


@lru_cache(maxsize=2)
def _load(path: str | None) -> dict[str, list[str]]:
    if path is None:
        text = files("ryokai").joinpath("data", "labelconfig.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text)


class LabelConfig:
    """Map raw SRL labels to canonical MEANT classes."""

    def __init__(self, path: str | None = None) -> None:
        raw = _load(path)
        # case-insensitive lookup: lowercase(alias) -> canonical class
        self._alias_to_class: dict[str, str] = {}
        for canonical, aliases in raw.items():
            for a in aliases:
                self._alias_to_class[a.lower()] = canonical
        self.classes: list[str] = list(raw.keys())

    def canonical(self, raw_label: str) -> str:
        """Return the canonical class for a raw SRL label, or 'OTHER'."""
        if raw_label is None:
            return "OTHER"
        return self._alias_to_class.get(raw_label.lower(), "OTHER")

    def __contains__(self, raw_label: str) -> bool:
        return raw_label.lower() in self._alias_to_class
