"""Minimal multilingual stopword loader (lazy, cached)."""
from __future__ import annotations

import string
from functools import lru_cache
from importlib.resources import files

import yaml

_PUNCT_SET = set(string.punctuation) | {
    "—", "–", "…", "•", "·", "“", "”", "‘", "’", "«", "»", "「", "」", "『", "』",
    "，", "。", "？", "！", "：", "；", "（", "）",
}


@lru_cache(maxsize=1)
def _load_all() -> dict[str, set[str]]:
    """Load stopwords. Uses BaseLoader so YAML's bool aliases (`no`, `yes`,
    `on`, `off`, `true`, `false`) and integer-looking tokens stay strings."""
    text = files("ryokai").joinpath("data", "stopwords.yaml").read_text(encoding="utf-8")
    raw: dict[str, list[str]] = yaml.load(text, Loader=yaml.BaseLoader)
    out: dict[str, set[str]] = {}
    for lang, words in raw.items():
        out[lang] = {str(w).lower() for w in words}
    return out


def stopwords_for(lang: str) -> set[str]:
    """Return the bundled stopword set for `lang` (empty set if unknown)."""
    return _load_all().get(lang, set())


def is_content_token(tok: str, lang: str) -> bool:
    """A token is a 'content word' if it isn't a stopword or pure punctuation."""
    t = tok.strip().lower()
    if not t:
        return False
    if all(c in _PUNCT_SET for c in t):
        return False
    return t not in stopwords_for(lang)
