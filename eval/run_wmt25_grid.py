"""Grid evaluation of ryokai configurations on the WMT25 GenMT human-eval data.

Grid:
  language pairs  : en-zh, en-ja
  modalities       : reference-based (refA) and reference-free (English src)
  variants per (LP, modality):
    A. aligner='sentence'  × sim_backend ∈ {qwen3-0.6b, embedding-gemma, jina-v2-base}
    B. aligner='bertscore' × ctx_backbone ∈ {xlm-r-large, qwen3-0.6b}
    C. aligner='itermax'   × ctx_backbone ∈ {xlm-r-large, qwen3-0.6b}
    D. srl=True            × sim_backend  ∈ {qwen3-0.6b, embedding-gemma}

Per configuration we score every (system, segment) pair and report:
  - system-level Pearson r and Spearman ρ
  - segment-level Kendall τ-b

Outputs a CSV (default `eval/wmt25_grid_results.csv`).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from ryokai import Ryokai
from ryokai.sim.contextual import ContextualTokenSimBackend
from ryokai.sim.embeddings import EmbeddingSimBackend


# --- grid definition --------------------------------------------------------

LANG_PAIRS = [
    ("en-zh", "en", "zh"),
    ("en-ja", "en", "ja"),
]

SENTENCE_SIM_BACKENDS = ["qwen3-0.6b", "embedding-gemma", "jina-v2-base"]
CONTEXTUAL_BACKBONES = ["xlm-r-large", "qwen3-0.6b"]
SRL_SIM_BACKENDS = ["qwen3-0.6b", "embedding-gemma"]


def grid_configs():
    """Yield (variant, label, ctor_kwargs) tuples."""
    for sim in SENTENCE_SIM_BACKENDS:
        yield "A", f"sentence/{sim}", dict(
            sim_backend=EmbeddingSimBackend(sim), aligner="sentence",
        )
    for ctx in CONTEXTUAL_BACKBONES:
        yield "B", f"bertscore/{ctx}", dict(
            sim_backend=ContextualTokenSimBackend(ctx), aligner="bertscore",
        )
    for ctx in CONTEXTUAL_BACKBONES:
        yield "C", f"itermax/{ctx}", dict(
            sim_backend=ContextualTokenSimBackend(ctx), aligner="itermax",
        )
    for sim in SRL_SIM_BACKENDS:
        yield "D", f"srl/{sim}", dict(
            sim_backend=EmbeddingSimBackend(sim),
        )


# --- WMT25 data loading -----------------------------------------------------

def load_wmt25(path: Path, lp: str) -> list[dict]:
    """Return list of records for one language pair.

    Each record: {doc_id, src, ref, systems: {name: hyp},
                  human: {name: float}}.
    Annotator-mean per (system) human score; refA excluded from the
    systems list (it's the gold reference, not an MT system to evaluate).
    """
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sample_sys = next(iter(r["scores"]))
            ann_prefix = r["scores"][sample_sys][0]["annotator"].split("_#_")[0]
            if ann_prefix != lp:
                continue
            tgt = r.get("tgt_text", {})
            ref = tgt.get("refA")
            systems_hyp = {s: h for s, h in tgt.items() if s != "refA"}
            human = {}
            for s, anns in r["scores"].items():
                if s == "refA" or s not in systems_hyp:
                    continue
                scores = [a.get("score") for a in anns if a.get("score") is not None]
                if scores:
                    human[s] = mean(scores)
            if not human or not ref or not r.get("src_text"):
                continue
            out.append({
                "doc_id": r["doc_id"],
                "src": r["src_text"],
                "ref": ref,
                "systems": {s: systems_hyp[s] for s in human if s in systems_hyp},
                "human": human,
            })
    return out


# --- scoring + correlation --------------------------------------------------

def score_config(
    records: list[dict],
    modality: str,
    variant: str,
    ctor_kwargs: dict,
    src_lang: str,
    tgt_lang: str,
) -> dict:
    """Score every (system, segment) pair under one configuration."""
    srl = (variant == "D")
    scorer = Ryokai(**ctor_kwargs)

    sys_scores: dict[str, list[float]] = {}
    sys_humans: dict[str, list[float]] = {}
    seg_pairs: list[tuple[float, float]] = []

    for r in records:
        for s, hyp in r["systems"].items():
            if s not in r["human"]:
                continue
            try:
                if modality == "ref":
                    sc = scorer.score(
                        reference=r["ref"], hypothesis=hyp,
                        target_lang=tgt_lang, srl=srl,
                    )
                else:  # source-based / reference-free
                    sc = scorer.score(
                        source=r["src"], hypothesis=hyp,
                        source_lang=src_lang, target_lang=tgt_lang, srl=srl,
                    )
            except Exception as e:  # noqa: BLE001
                print(f"    error scoring {s} on {r['doc_id']}: {e!r}")
                continue
            sys_scores.setdefault(s, []).append(sc.f1)
            sys_humans.setdefault(s, []).append(r["human"][s])
            seg_pairs.append((sc.f1, r["human"][s]))

    out = {"n_systems": len(sys_scores), "n_segments": len(seg_pairs)}
    # system-level: mean per system, then correlate across systems
    if len(sys_scores) >= 3:
        sys_means = {s: float(np.mean(v)) for s, v in sys_scores.items()}
        hum_means = {s: float(np.mean(sys_humans[s])) for s in sys_scores}
        keys = sorted(sys_means)
        rx = [sys_means[k] for k in keys]
        ry = [hum_means[k] for k in keys]
        try:
            out["sys_pearson"] = float(pearsonr(rx, ry).statistic)
            out["sys_spearman"] = float(spearmanr(rx, ry).statistic)
        except Exception:
            out["sys_pearson"] = float("nan")
            out["sys_spearman"] = float("nan")
    else:
        out["sys_pearson"] = float("nan")
        out["sys_spearman"] = float("nan")

    # segment-level Kendall tau-b
    if len(seg_pairs) >= 3 and len(set(p[0] for p in seg_pairs)) > 1:
        try:
            kx = [p[0] for p in seg_pairs]
            ky = [p[1] for p in seg_pairs]
            out["seg_kendall_b"] = float(kendalltau(kx, ky, variant="b").statistic)
        except Exception:
            out["seg_kendall_b"] = float("nan")
    else:
        out["seg_kendall_b"] = float("nan")
    return out


# --- main -------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="/Users/alvas/Downloads/wmt25-genmt-humeval.jsonl")
    p.add_argument("--out",  default="eval/wmt25_grid_results.csv")
    p.add_argument(
        "--subsample", type=int, default=0,
        help="Random-subsample to this many segments per LP (0 = full).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--lps", nargs="+", default=None,
        help="Restrict to a subset of LPs (e.g. --lps en-zh).",
    )
    p.add_argument(
        "--modalities", nargs="+", default=["ref", "src"],
        choices=["ref", "src"],
    )
    p.add_argument(
        "--variants", nargs="+", default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
    )
    args = p.parse_args()

    random.seed(args.seed)
    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected_lps = [(lp, sl, tl) for lp, sl, tl in LANG_PAIRS
                    if args.lps is None or lp in args.lps]
    configs = [c for c in grid_configs() if c[0] in args.variants]
    total_runs = len(selected_lps) * len(args.modalities) * len(configs)
    print(f"grid: {len(selected_lps)} LPs × {len(args.modalities)} modalities "
          f"× {len(configs)} configs = {total_runs} runs", flush=True)

    fieldnames = [
        "lp", "modality", "variant", "config",
        "n_systems", "n_segments",
        "sys_pearson", "sys_spearman", "seg_kendall_b", "wall_clock_s",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for lp, src_lang, tgt_lang in selected_lps:
            print(f"\n=== loading {lp} ===", flush=True)
            recs = load_wmt25(data_path, lp)
            print(f"    {len(recs)} records, "
                  f"{sum(len(r['systems']) for r in recs)} (sys, seg) pairs",
                  flush=True)
            if args.subsample and len(recs) > args.subsample:
                recs = random.sample(recs, args.subsample)
                print(f"    subsampled to {len(recs)}", flush=True)

            for modality in args.modalities:
                for variant, label, kwargs in configs:
                    tag = f"[{lp} / {modality} / {label}]"
                    print(f"\n  {tag} starting ...", flush=True)
                    t0 = time.time()
                    res = score_config(
                        recs, modality, variant, kwargs,
                        src_lang, tgt_lang,
                    )
                    dt = time.time() - t0
                    row = dict(
                        lp=lp, modality=modality, variant=variant,
                        config=label, wall_clock_s=round(dt, 1),
                        **res,
                    )
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"  {tag} done in {dt:.0f}s: "
                        f"sys_pearson={res['sys_pearson']:+.3f} "
                        f"sys_spearman={res['sys_spearman']:+.3f} "
                        f"seg_kendall_b={res['seg_kendall_b']:+.3f} "
                        f"(n_sys={res['n_systems']}, n_seg={res['n_segments']})",
                        flush=True,
                    )

    print(f"\nresults written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
