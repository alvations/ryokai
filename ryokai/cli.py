"""ryokai CLI — score a parallel hyp file vs reference or source file."""
from __future__ import annotations

import argparse
import sys

from . import Ryokai, SUPPORTED_LANGS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ryokai",
        description=(
            "Score MT output. Pass --ref for reference-based scoring "
            "(MEANT / WOLVESAAR / YiSi-1 / SimAlign style), or --src "
            "for reference-free scoring (XMEANT / YiSi-2 / Doc-embedding "
            "adequacy)."
        ),
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ref", help="Path to reference text (one segment per line).")
    group.add_argument("--src", help="Path to source text (one segment per line).")
    p.add_argument("--hyp", required=True, help="Path to hypothesis / MT output (one segment per line).")
    p.add_argument(
        "--target-lang", required=True, choices=sorted(SUPPORTED_LANGS),
        help="Two-letter language code of the hypothesis.",
    )
    p.add_argument(
        "--source-lang", choices=sorted(SUPPORTED_LANGS),
        help="Language of --src. Defaults to --target-lang. Ignored if --ref is used.",
    )
    p.add_argument(
        "--srl", action="store_true",
        help="Use the frame-based MEANT 2.0 / XMEANT scorer. Default: word alignment + embedding.",
    )
    p.add_argument(
        "--aligner", default="sentence",
        choices=["sentence", "hungarian", "argmax", "itermax", "mai"],
        help="No-SRL aligner. Ignored when --srl is set.",
    )
    p.add_argument(
        "--sim-model", default=None,
        help="Override the multilingual embedding model id or preset.",
    )
    p.add_argument(
        "--no-corpus-avg", action="store_true",
        help="Print one F1 per line instead of corpus mean.",
    )
    args = p.parse_args(argv)

    hyps = [line.rstrip("\n") for line in open(args.hyp, encoding="utf-8")]
    if args.ref:
        others = [line.rstrip("\n") for line in open(args.ref, encoding="utf-8")]
        kind = "references"
    else:
        others = [line.rstrip("\n") for line in open(args.src, encoding="utf-8")]
        kind = "sources"
    if len(others) != len(hyps):
        print(
            f"{kind} and hypotheses line counts differ: {len(others)} vs {len(hyps)}",
            file=sys.stderr,
        )
        return 2

    from .sim.embeddings import EmbeddingSimBackend
    sim = EmbeddingSimBackend(args.sim_model) if args.sim_model else None
    scorer = Ryokai(sim_backend=sim, aligner=args.aligner)
    scores = scorer.score_corpus(
        hypotheses=hyps,
        target_lang=args.target_lang,
        sources=others if args.src else None,
        references=others if args.ref else None,
        source_lang=args.source_lang,
        srl=args.srl,
    )

    if args.no_corpus_avg:
        for s in scores:
            print(f"{s.f1:.4f}")
    else:
        mean = sum(s.f1 for s in scores) / len(scores) if scores else 0.0
        print(f"ryokai = {mean:.4f}  (n={len(scores)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
