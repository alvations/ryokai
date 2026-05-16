"""Heuristic SRL using a multilingual HF POS tagger.

Pure-transformers backend — no stanza, no spaCy, no separate parser.

Approach (works the same across all 13 MEANT languages):

  1. Tag every token with universal POS via a single multilingual XLM-R
     POS model (default: `wietsedv/xlm-roberta-base-ft-udpos28`, covering
     28 UD languages including all of MEANT's 13).
  2. Predicates = maximal contiguous VERB/AUX spans.
  3. For each predicate:
       - the most recent NOUN/PROPN/PRON phrase to its **left**  -> ARG0
       - the next       NOUN/PROPN/PRON phrase to its **right** -> ARG1
       - any further NOUN/PROPN phrase further right (or oblique) -> ARG2
       - NUM/ADV/PART tokens adjacent to the predicate           -> ARGM-*
                                                                    (TMP/LOC/MNR
                                                                    via token-text
                                                                    heuristics)
  4. Emit a Frame per predicate.

This is intentionally shallow — MEANT scoring is robust because what
matters is **consistent treatment of ref and hyp**, not absolute SRL
correctness. Users who want PropBank-grade frames should plug in
`HFTokenClassifierSRLBackend` with a real SRL model.
"""
from __future__ import annotations

from functools import lru_cache

from ..graph import Argument, Frame, SRLGraph
from . import SRLBackend

# Per-language XLM-R UDPOS28 models (Wietse de Vries et al., ACL 2022).
# The pattern `{lang}` is filled in per-call from the `lang` arg to .parse().
# Coverage matches `_langs._EXTRA_LANGS` + `MEANT_LANGS` (~38 langs total).
DEFAULT_POS_MODEL = "wietsedv/xlm-roberta-base-ft-udpos28-{lang}"

_PRED_POS = {"VERB", "AUX"}
_ARG_POS = {"NOUN", "PROPN", "PRON"}
_MOD_POS = {"ADV", "PART", "NUM"}

# rough textual cues used to refine ARGM-* (case-insensitive substring match)
# kept short and language-agnostic — these are best-effort hints, not exhaustive.
_TMP_CUES = {
    "yesterday", "today", "tomorrow", "now", "soon", "later", "ago",
    "morgen", "gestern", "heute", "jetzt", "bald",
    "hier", "aujourd'hui", "demain", "maintenant",
    "ayer", "hoy", "mañana", "ahora", "luego",
    "вчера", "сегодня", "завтра", "сейчас",
    "昨天", "今天", "明天", "现在",
    "dün", "bugün", "yarın", "şimdi",
}
_NEG_CUES = {
    "not", "n't", "no", "never",
    "nicht", "kein", "keine",
    "ne", "pas",
    "no", "nunca",
    "ne", "не",
    "yok", "değil",
    "नहीं", "ना",
    "不", "没", "未",
}


@lru_cache(maxsize=4)
def _load_pos_pipeline(model_id: str):
    from transformers import pipeline
    return pipeline(
        task="token-classification",
        model=model_id,
        aggregation_strategy="none",
    )


def _tokens_with_pos(sentence: str, model_id: str) -> list[tuple[str, str]]:
    """Run POS tagger and return [(word, upos), ...]."""
    pipe = _load_pos_pipeline(model_id)
    raw = pipe(sentence)
    out: list[tuple[str, str]] = []
    cur_word, cur_pos = "", ""
    for tok in raw:
        word = tok.get("word", "")
        pos = tok.get("entity", tok.get("entity_group", "X"))
        # XLM-R subwords start with ▁ — concatenate continuations
        if word.startswith("▁"):
            if cur_word:
                out.append((cur_word, cur_pos))
            cur_word = word.lstrip("▁")
            cur_pos = pos
        else:
            cur_word += word
    if cur_word:
        out.append((cur_word, cur_pos))
    return out


def _find_pred_spans(tokens: list[tuple[str, str]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i][1] in _PRED_POS:
            j = i
            while j < n and tokens[j][1] in _PRED_POS:
                j += 1
            spans.append((i, j))  # half-open
            i = j
        else:
            i += 1
    return spans


def _find_np_span(
    tokens: list[tuple[str, str]],
    start: int,
    direction: int,
    bound: int,
) -> tuple[int, int] | None:
    """Search outward from `start` for the nearest NOUN-headed span.
    direction = +1 (right) or -1 (left). Returns (s, e) half-open or None."""
    i = start
    while 0 <= i < len(tokens) and (direction > 0 and i < bound) or (
        direction < 0 and i > bound
    ):
        if tokens[i][1] in _ARG_POS:
            # grow contiguous run of N/Adj/Det around it
            s = i
            while s - 1 >= 0 and tokens[s - 1][1] in _ARG_POS | {"DET", "ADJ"}:
                s -= 1
            e = i + 1
            while e < len(tokens) and tokens[e][1] in _ARG_POS | {"ADJ"}:
                e += 1
            return (s, e)
        i += direction
    return None


def _span_text(tokens: list[tuple[str, str]], s: int, e: int) -> str:
    return " ".join(w for w, _ in tokens[s:e])


def _argm_label(text: str) -> str | None:
    lo = text.lower()
    if any(c in lo for c in _TMP_CUES):
        return "ARGM-TMP"
    if any(c in lo for c in _NEG_CUES):
        return "ARGM-NEG"
    return None


class HFPOSHeuristicSRLBackend(SRLBackend):
    """Pseudo-SRL using a multilingual HF POS tagger only.

    Default model id is the Wietse de Vries XLM-R UDPOS28 family
    (`wietsedv/xlm-roberta-base-ft-udpos28-{lang}`). The `{lang}` token
    is filled per-call from the `lang` argument of `.parse()`, so the
    correct per-language model is loaded automatically.

    Pass an explicit string without `{lang}` to pin a single model:

        HFPOSHeuristicSRLBackend(
            pos_model="wietsedv/xlm-roberta-base-ft-udpos28-en",
        )
    """

    def __init__(self, pos_model: str = DEFAULT_POS_MODEL) -> None:
        self.pos_model = pos_model

    def _resolve_model_id(self, lang: str) -> str:
        if "{lang}" in self.pos_model:
            return self.pos_model.format(lang=lang)
        return self.pos_model

    def parse(self, sentence: str, lang: str) -> SRLGraph:
        tokens = _tokens_with_pos(sentence, self._resolve_model_id(lang))
        graph = SRLGraph(sentence=sentence)
        if not tokens:
            return graph

        pred_spans = _find_pred_spans(tokens)
        if not pred_spans:
            return graph

        for k, (ps, pe) in enumerate(pred_spans):
            pred_text = _span_text(tokens, ps, pe)
            frame = Frame(predicate=pred_text)

            # search bounds: don't cross into another predicate's territory
            left_bound = pred_spans[k - 1][1] if k > 0 else -1
            right_bound = pred_spans[k + 1][0] if k + 1 < len(pred_spans) else len(tokens)

            left_np = _find_np_span(tokens, ps - 1, -1, left_bound)
            if left_np is not None:
                txt = _span_text(tokens, *left_np)
                frame.arguments.append(Argument(label="", raw_label="ARG0", text=txt))

            right_np = _find_np_span(tokens, pe, +1, right_bound)
            if right_np is not None:
                txt = _span_text(tokens, *right_np)
                frame.arguments.append(Argument(label="", raw_label="ARG1", text=txt))
                # try for an ARG2 further right
                far_np = _find_np_span(tokens, right_np[1], +1, right_bound)
                if far_np is not None:
                    txt2 = _span_text(tokens, *far_np)
                    frame.arguments.append(Argument(label="", raw_label="ARG2", text=txt2))

            # ARGM hints: any ADV/PART/NUM in the predicate's neighbourhood
            for i in range(max(left_bound + 1, ps - 3), min(right_bound, pe + 3)):
                if 0 <= i < len(tokens) and tokens[i][1] in _MOD_POS:
                    raw = _argm_label(tokens[i][0]) or "ARGM-MNR"
                    frame.arguments.append(
                        Argument(label="", raw_label=raw, text=tokens[i][0])
                    )

            if frame.arguments:
                graph.frames.append(frame)
        return graph
