"""ryokai CLI: score a parallel ref/hyp text file pair."""
from __future__ import annotations

import argparse
import sys

from . import MEANT, SUPPORTED_LANGS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ryokai",
        description="Score MT output with MEANT (semantic-frame eval metric).",
    )
    p.add_argument("--ref", required=True, help="Path to reference text (one segment per line).")
    p.add_argument("--hyp", required=True, help="Path to hypothesis text (one segment per line).")
    p.add_argument(
        "--lang",
        required=True,
        choices=sorted(SUPPORTED_LANGS),
        help="Two-letter language code.",
    )
    p.add_argument(
        "--sim-model",
        default=None,
        help="Override the multilingual embedding model (default: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).",
    )
    p.add_argument(
        "--no-corpus-avg",
        action="store_true",
        help="Print one F1 per line instead of corpus mean.",
    )
    args = p.parse_args(argv)

    refs = [line.rstrip("\n") for line in open(args.ref, encoding="utf-8")]
    hyps = [line.rstrip("\n") for line in open(args.hyp, encoding="utf-8")]
    if len(refs) != len(hyps):
        print(f"refs and hyps line counts differ: {len(refs)} vs {len(hyps)}", file=sys.stderr)
        return 2

    from .sim.embeddings import EmbeddingSimBackend
    sim = EmbeddingSimBackend(args.sim_model) if args.sim_model else None
    meant = MEANT(lang=args.lang, sim_backend=sim)
    scores = meant.score_corpus(refs, hyps)

    if args.no_corpus_avg:
        for s in scores:
            print(f"{s.f1:.4f}")
    else:
        mean = sum(s.f1 for s in scores) / len(scores) if scores else 0.0
        print(f"MEANT = {mean:.4f}  (n={len(scores)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
