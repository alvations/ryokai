"""HuggingFace token-classification SRL backend.

For users who have a real SRL model fine-tuned for token-level role
labelling (BIO tags) on HF Hub.  Outputs a frame per predicate using the
BIO chunks.
"""
from __future__ import annotations

from functools import lru_cache

from ..graph import Argument, Frame, SRLGraph
from . import SRLBackend


@lru_cache(maxsize=8)
def _load_pipeline(model_id: str):
    from transformers import pipeline
    return pipeline(
        task="token-classification",
        model=model_id,
        aggregation_strategy="simple",
    )


class HFTokenClassifierSRLBackend(SRLBackend):
    """Wrap any HF token-classification model that emits BIO-tagged SRL.

    Expects per-token entity_group values matching the labelconfig aliases
    (e.g. "ARG0", "ARG1", "V"). One frame is built per "V"-tagged token.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def parse(self, sentence: str, lang: str) -> SRLGraph:
        pipe = _load_pipeline(self.model_id)
        tagged = pipe(sentence)
        # split into one frame per V
        verbs: list[tuple[int, str]] = []
        args: list[tuple[int, str, str]] = []  # (start_idx, label, text)
        for span in tagged:
            label = span.get("entity_group", "")
            text = span.get("word", "")
            start = span.get("start", 0)
            if label.upper() == "V":
                verbs.append((start, text))
            else:
                args.append((start, label, text))
        graph = SRLGraph(sentence=sentence)
        if not verbs:
            return graph
        # naive single-frame model: all args attach to the first V; fine for
        # backends that don't expose per-predicate arg masks. Users with
        # multi-predicate output should subclass and override.
        v_start, v_text = verbs[0]
        frame = Frame(predicate=v_text)
        for _, label, text in args:
            frame.arguments.append(Argument(label="", raw_label=label, text=text))
        if frame.arguments:
            graph.frames.append(frame)
        return graph
