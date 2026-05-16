# ryokai 了解

> *Ryokai* (了解, "understood / got it") — a unified Python library for **semantic machine-translation evaluation**, combining the strengths of MEANT, XMEANT, YiSi, WOLVESAAR, and SimAlign behind one clean API on top of modern multilingual embeddings.

Pure PyTorch + HuggingFace `transformers` — no Stanza, no spaCy, no external parsers. Two HF models cover **all 13 MEANT languages** (`en`, `de`, `fr`, `es`, `cs`, `fi`, `hi`, `lv`, `pl`, `ro`, `ru`, `tr`, `zh`) in a single install:

- POS / shallow SRL: [`wietsedv/xlm-roberta-base-ft-udpos28`](https://huggingface.co/wietsedv/xlm-roberta-base-ft-udpos28) — ~1.1 GB once.
- Multilingual embeddings: [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — ~110 MB once, used for both **same-language** and **cross-language** similarity.

## What's inside

Ryokai is a single-import home for several MT-eval techniques that have historically lived in separate codebases (or, in MEANT's case, only as C++ binaries):

| Technique | Year | What it brings | ryokai entry point |
| --------- | ---- | -------------- | ------------------ |
| **MEANT 2.0** (Lo) | 2017 | Semantic-frame F-score: predicates + role fillers aligned via Hungarian matching | `MEANT(lang).score(ref, hyp)` |
| **XMEANT** (Lo et al.) | 2014 | Reference-free MEANT — score MT against source directly | `MEANT(lang).score_xlingual(src, hyp, source_lang)` |
| **YiSi-1 / YiSi-2** (Lo) | 2019 | Embedding-based scoring, with / without SRL, with / without reference | `MEANT(lang, use_srl=False)` |
| **WOLVESAAR** (Bechara, Gupta, Tan et al.) | 2016 | Monolingual content-word alignment + sentence vector composition | `score_nosrl(ref, hyp, content_only=True, aggregation="harmonic")` |
| **Sultan et al.** | 2014 | Contextual word alignment with exact-match prior + threshold | `ContextualTokenSimBackend` with `aligner="hungarian"` |
| **SimAlign** (Jalili Sabet et al.) | 2020 | ArgMax / IterMax / MAI word aligners over multilingual encoders | `aligner="argmax"` / `"itermax"` / `"mai"` |
| **AER** (Och & Ney) | 2003 | Standard evaluation harness for word aligners | `ryokai.eval.aer(...)` / `evaluate_aligner(...)` |

The role-label canonicalisation (`ryokai/data/labelconfig.yaml`) collapses PropBank / Chinese PropBank / AnCora / reference-and-continuation tags into one ontology so the same scorer works across languages.

## Install

```bash
pip install ryokai
```

## Quickstart

```python
from ryokai import MEANT

meant = MEANT(lang="en")

score = meant.score(
    reference="The cat sat on the mat yesterday.",
    hypothesis="A cat was sitting on the mat yesterday.",
)
print(score)
# MEANTScore(precision=0.74, recall=0.82, f1=0.78, n_frames_ref=1, n_frames_hyp=1, n_aligned_predicates=1)

float(score)   # 0.78  — convenient shortcut
```

Corpus-level:

```python
scores = meant.score_corpus(refs, hyps)
corpus_meant = sum(s.f1 for s in scores) / len(scores)
```

Any of the 13 languages — same call, just change `lang`:

```python
MEANT("zh").score("猫坐在垫子上。", "一只猫在垫子上。")
MEANT("hi").score("बिल्ली चटाई पर बैठी।", "एक बिल्ली चटाई पर बैठी थी।")
```

## Variants

### Default — full MEANT

```python
MEANT(lang="en").score(reference, hypothesis)
```

### No-SRL fallback (YiSi-1 / WOLVESAAR / SimAlign style)

Skip SRL entirely and score by word alignment + embedding similarity. Four aligners are available, trading off speed vs. faithfulness to the literature:

| `aligner=` | Backbone | Matching | Notes |
| ---------- | -------- | -------- | ----- |
| `"sentence"` *(default)* | multilingual MiniLM (already loaded) | Hungarian | Fast, no extra download. Tokens embedded out of context — coarse approximation. |
| `"hungarian"` | XLM-RoBERTa contextual | scipy 1-to-1 Hungarian | Sultan et al. (2014) style: contextual vectors + exact-match prior + cosine threshold. |
| `"argmax"`    | XLM-RoBERTa contextual | bidirectional argmax intersection | SimAlign Inter: each src picks best tgt; keep only mutual best. Allows many-to-many. |
| `"itermax"`   | XLM-RoBERTa contextual | argmax intersection + iterative growth | SimAlign IterMax — **recommended quality**. Adds back unaligned words above the threshold. |
| `"mai"`       | XLM-RoBERTa contextual | union of `hungarian` ∪ `argmax` ∪ `itermax` | SimAlign `mai` combinator — highest recall, useful as an upper-bound or for sensitivity analysis. |

For a no-GPU / no-forward-pass static backend (`fastText`-style):

```python
from ryokai import MEANT, StaticEmbeddingSimBackend
from ryokai.scorer import score_nosrl

# uses only XLM-R's input embedding layer — no transformer forward pass
score_nosrl(ref, hyp, aligner="itermax",
            sim_backend=StaticEmbeddingSimBackend("xlm-r-base"))

# or actual fastText vectors via gensim (optional install)
StaticEmbeddingSimBackend(fasttext_path="cc.en.300.vec")
```

```python
# Fast — no extra model download
MEANT(lang="en", use_srl=False).score(ref, hyp)

# Sultan et al. style: contextual encoder + threshold + exact-match prior
MEANT(lang="en", use_srl=False, aligner="hungarian").score(ref, hyp)

# SimAlign IterMax — best F1 in the SimAlign paper
MEANT(lang="en", use_srl=False, aligner="itermax",
      content_only=True, threshold=0.5).score(ref, hyp)
```

Extra flags for all no-SRL modes:

- `content_only=True` — drop stopwords + punctuation before alignment (Sultan et al. / WOLVESAAR `prop_harmonic` feature). Per-language stopwords bundled for all 13 MEANT languages in `ryokai/data/stopwords.yaml`.
- `aggregation="harmonic"` — explicitly request the WOLVESAAR harmonic mean of per-sentence aligned-content-word proportions (mathematically the same as F1, name kept for clarity).
- `threshold=0.5` — drop alignments with cosine below this (contextual aligners only).
- `exact_match_shortcut=True` — case-insensitive surface-form equality boosts a pair's similarity to 1.0 (contextual aligners only). This is Sultan et al.'s "lexical prior" cascade in modern dress.

## Swapping the embedding backbone

Both backends (`EmbeddingSimBackend` for sentence/argument similarity, `ContextualTokenSimBackend` for word alignment) accept any HuggingFace model id, plus a handful of shorthand presets for popular modern choices:

```python
from ryokai import MEANT, EmbeddingSimBackend
from ryokai.sim.contextual import ContextualTokenSimBackend

# Upgrade the sentence-level backbone (used for predicate / arg similarity)
meant = MEANT(
    lang="en",
    sim_backend=EmbeddingSimBackend("qwen3-0.6b"),       # or "jina-v3", "mpnet", or any HF id
)

# Upgrade the contextual word aligner backbone (used by aligner=hungarian/argmax/itermax)
from ryokai.scorer import score_nosrl
score_nosrl(
    reference, hypothesis,
    aligner="itermax",
    # ContextualTokenSimBackend is instantiated internally — pass a custom one via:
    # ryokai.sim.contextual.ContextualTokenSimBackend(model_id="qwen3-0.6b")
)
```

| Preset key            | Model                                                       | Year  | Backend |
| --------------------- | ----------------------------------------------------------- | ----- | ------- |
| `minilm`              | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 2021  | sentence (default) |
| `mpnet`               | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 2021  | sentence |
| `mbert`               | bert-base-multilingual-cased                                | 2019  | contextual |
| `xlm-r-base`          | xlm-roberta-base                                            | 2020  | contextual (default) |
| `xlm-r-large`         | xlm-roberta-large                                           | 2020  | contextual |
| `mdeberta-v3`         | microsoft/mdeberta-v3-base                                  | 2022  | contextual |
| `me5-large`           | intfloat/multilingual-e5-large                              | 2024  | sentence |
| `me5-large-instruct`  | intfloat/multilingual-e5-large-instruct                     | 2024  | sentence |
| `bge-m3`              | BAAI/bge-m3                                                 | 2024  | sentence + contextual |
| `mxbai-large`         | mixedbread-ai/mxbai-embed-large-v1                          | 2024  | sentence |
| `snowflake-arctic-v2` | Snowflake/snowflake-arctic-embed-l-v2.0                     | 2024  | sentence + contextual |
| `nomic-v2`            | nomic-ai/nomic-embed-text-v2-moe                            | 2025  | sentence |
| `jina-v2-base`        | jinaai/jina-embeddings-v2-base-en                           | 2023  | sentence + contextual |
| `jina-v3`             | jinaai/jina-embeddings-v3                                   | 2024  | sentence + contextual |
| `embedding-gemma`     | google/embeddinggemma-300m                                  | 2025  | sentence + contextual |
| **`qwen3-0.6b`**      | Qwen/Qwen3-Embedding-0.6B                                   | **2025** | sentence + contextual |
| `qwen3-4b`            | Qwen/Qwen3-Embedding-4B                                     | 2025  | sentence + contextual |
| `qwen3-8b`            | Qwen/Qwen3-Embedding-8B                                     | 2025  | sentence + contextual |
| **`nemotron-8b`**     | nvidia/llama-embed-nemotron-8b                              | **2025** (#1 MMTEB Oct 2025) | sentence + contextual |

Any non-preset string is passed straight through to `transformers.AutoModel` / `sentence_transformers.SentenceTransformer` (with `trust_remote_code=True` so models like Jina v3 / Nomic v2 that ship custom code work out of the box).

### SimAlign-inspired extras on the contextual backend

```python
from ryokai.sim.contextual import ContextualTokenSimBackend

# pick a middle layer (SimAlign found layer 8 of mBERT-base / XLM-R-base
# beats the last layer for word alignment)
be = ContextualTokenSimBackend("xlm-r-base", layer=8)

# drop alignments where word positions differ by more than 30% of sentence length
be = ContextualTokenSimBackend("qwen3-0.6b", max_distortion=0.3)
```

These plug into any of the `aligner="hungarian"/"argmax"/"itermax"/"mai"` modes via the backend you pass in.

## Evaluating the aligners themselves (AER)

For benchmarking ryokai's aligners against a gold word-alignment dataset (WPT, Hansards, Europarl, etc.), `ryokai.eval` exposes the standard Och & Ney (2003) AER metric and convenience helpers:

```python
from ryokai import ContextualTokenSimBackend
from ryokai.eval import aer, evaluate_aligner

# Single-pair AER
predicted = [(0, 0, 0.9), (1, 1, 0.85)]   # (i, j, similarity) tuples are OK
sure = {(0, 0), (1, 1)}
print(aer(predicted, sure))                # 0.0 — perfect

# Sweep an aligner over a corpus and report aggregate AER / P / R / F1
backend = ContextualTokenSimBackend("xlm-r-base", layer=8)
res = evaluate_aligner(
    backend,
    examples=[
        ("the cat sat", "le chat s'est assis",
         {(0, 0), (1, 1), (2, 3)},      # sure
         None),                          # possible (None ⇒ same as sure)
        ...,
    ],
    method="itermax",
    threshold=0.5,
)
print(res)   # {'n': 100, 'aer': 0.18, 'precision': 0.84, 'recall': 0.79, 'f1': 0.81}
```

The metric helpers (`aer`, `precision`, `recall`, `f1`) all accept either 2-tuple `(i, j)` or 3-tuple `(i, j, similarity)` alignment pairs, so output from `ContextualTokenSimBackend.align()` / `StaticEmbeddingSimBackend.align()` plugs in directly.

### Reference-free (XMEANT / YiSi-2 style)

Score MT directly against the **source** sentence, no reference needed. Works because the embedding backbone is multilingual.

```python
meant = MEANT(lang="zh")   # MT output language
meant.score_xlingual(
    source="The cat sat on the mat yesterday.",
    hypothesis="猫昨天坐在垫子上。",
    source_lang="en",
)
```

## CLI

```bash
ryokai --ref ref.txt --hyp hyp.txt --lang en
# MEANT = 0.71  (n=2000)

ryokai --ref ref.txt --hyp hyp.txt --lang en --no-corpus-avg > per_sentence.f1
```

## Architecture

```
ryokai/
├── __init__.py             # MEANT facade + score_xlingual()
├── labelconfig.py          # load + apply role aliasing
├── data/labelconfig.yaml   # ported from meant.labelconfig
├── graph.py                # SRLGraph, Frame, Argument dataclasses
├── match.py                # scipy.linear_sum_assignment wrapper
├── scorer.py               # weighted F-score (frame-based + no-SRL)
├── srl/
│   ├── __init__.py         # SRLBackend ABC
│   ├── hf_pos.py           # default: HF POS-heuristic SRL (one XLM-R model)
│   └── hf_backend.py       # plug-in: any HF token-classification SRL model
├── sim/
│   └── embeddings.py       # default: multilingual MiniLM cosine sim
└── cli.py                  # argparse entry point
```

The **default SRL backend** (`HFPOSHeuristicSRLBackend`) is intentionally shallow — it uses a multilingual XLM-R POS tagger and identifies predicates as `VERB`/`AUX` spans with the nearest noun phrases on each side becoming `ARG0`/`ARG1`/`ARG2`. For MT eval, what matters is **consistent treatment of ref and hyp**, not absolute SRL correctness.

For PropBank-grade SRL on a specific language, plug in a real SRL model:

```python
from ryokai import MEANT, HFTokenClassifierSRLBackend

meant = MEANT(
    lang="en",
    srl_backend=HFTokenClassifierSRLBackend("liaad/srl-en_xlmr-base"),
)
```

## Custom weights

`MEANT(lang=…, weights={...})` accepts any `{role_class: float}` dict. Defaults give equal weight to V/A0/A1, 0.8 to A2/NEG, 0.6 to TMP/LOC, less to modifiers. Setting a weight to 0 ignores that role class entirely.

## Running the tests

Fast tests (label aliasing, scorer arithmetic, matching, no-SRL scorer) — no model downloads:

```bash
pytest -m "not slow"     # 30 tests, < 5s
```

Full multilingual end-to-end (~1.2 GB of model downloads on first run, then cached):

```bash
pytest -m slow           # 53 tests, all 13 langs × MEANT, no-SRL, and an XMEANT case
```

## Citations

If you use ryokai in research, please cite the original metric papers alongside this port:

```bibtex
@inproceedings{lo-wu-2011-meant,
  title     = {{MEANT}: An inexpensive, high-accuracy, semi-automatic metric for evaluating translation utility based on semantic roles},
  author    = {Lo, Chi-kiu and Wu, Dekai},
  booktitle = {Proceedings of ACL},
  year      = {2011},
}

@inproceedings{lo-2017-meant-2,
  title     = {{MEANT} 2.0: Accurate semantic {MT} evaluation for any output language},
  author    = {Lo, Chi-kiu},
  booktitle = {Proceedings of WMT},
  year      = {2017},
}

@inproceedings{lo-etal-2014-xmeant,
  title     = {{XMEANT}: Better semantic {MT} evaluation without reference translations},
  author    = {Lo, Chi-kiu and Beloucif, Meriem and Saers, Markus and Wu, Dekai},
  booktitle = {Proceedings of ACL (Short Papers)},
  pages     = {765--771},
  year      = {2014},
  url       = {https://aclanthology.org/P14-2124/},
}

@inproceedings{lo-2019-yisi,
  title     = {{YiSi} -- a Unified Semantic {MT} Quality Evaluation and Estimation Metric for Languages with Different Levels of Available Resources},
  author    = {Lo, Chi-kiu},
  booktitle = {Proceedings of WMT},
  year      = {2019},
}
```

The no-SRL aligners draw on:

```bibtex
@inproceedings{sultan-etal-2014-back,
  title     = {Back to Basics for Monolingual Alignment: Exploiting Word Similarity and Contextual Evidence},
  author    = {Sultan, Md Arafat and Bethard, Steven and Sumner, Tamara},
  journal   = {Transactions of the Association for Computational Linguistics},
  year      = {2014},
}

@inproceedings{bechara-etal-2016-wolvesaar,
  title     = {{WOLVESAAR} at {S}em{E}val-2016 Task 1: Replicating the Success of Monolingual Word Alignment and Neural Embeddings for Semantic Textual Similarity},
  author    = {Bechara, Hannah and Gupta, Rohit and Tan, Liling and Or{\u{a}}san, Constantin and Mitkov, Ruslan and van Genabith, Josef},
  booktitle = {Proceedings of SemEval-2016},
  pages     = {634--639},
  year      = {2016},
  url       = {https://aclanthology.org/S16-1096/},
}

@inproceedings{jalili-sabet-etal-2020-simalign,
  title     = {{S}im{A}lign: High Quality Word Alignments without Parallel Training Data using Static and Contextualized Embeddings},
  author    = {Jalili Sabet, Masoud and Dufter, Philipp and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
  booktitle = {Findings of EMNLP 2020},
  year      = {2020},
}
```

## License

MIT — see `LICENSE`.
