# ryokai 了解

> *Ryokai* (了解, "understood / got it") — a unified Python library for **semantic machine-translation evaluation**, combining the strengths of MEANT 2.0, XMEANT, YiSi-1/2, WOLVESAAR, and SimAlign behind one clean API on top of modern multilingual embeddings.

Pure PyTorch + HuggingFace `transformers` — no Stanza, no spaCy, no external parsers. Two HF models cover **all 13 supported languages** (`en`, `de`, `fr`, `es`, `cs`, `fi`, `hi`, `lv`, `pl`, `ro`, `ru`, `tr`, `zh`) in a single install:

- POS / shallow SRL: [`wietsedv/xlm-roberta-base-ft-udpos28`](https://huggingface.co/wietsedv/xlm-roberta-base-ft-udpos28) — ~1.1 GB, downloaded once.
- Multilingual embeddings: [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — ~110 MB, used for both **same-language** and **cross-language** similarity.

Both are one-line swappable for any modern multilingual encoder (Qwen3-Embedding, Jina v3, BGE-M3, Nemotron-8B…) — see [Embedding backbones](#embedding-backbones) below.

## What's inside

| Technique | Year | What it brings | Entry point |
| --------- | ---- | -------------- | ----------- |
| **MEANT 2.0** (Lo) | 2017 | Semantic-frame F-score: predicates + role fillers aligned via Hungarian matching | `Ryokai(lang).score(ref, hyp)` |
| **XMEANT** (Lo et al.) | 2014 | Reference-free MEANT — score MT against source directly | `Ryokai(lang).score_xlingual(src, hyp, source_lang)` |
| **YiSi-1 / YiSi-2** (Lo) | 2019 | Embedding-based scoring, with / without SRL, with / without reference | `Ryokai(lang, use_srl=False)` |
| **WOLVESAAR** (Bechara, Gupta, Tan et al.) | 2016 | Monolingual content-word alignment + sentence vector composition | `score_nosrl(..., content_only=True, aggregation="harmonic")` |
| **Sultan et al.** | 2014 | Contextual word alignment with exact-match prior + cosine threshold | `aligner="hungarian"` on the contextual backend |
| **SimAlign** (Jalili Sabet et al.) | 2020 | ArgMax / IterMax / MAI word aligners over multilingual encoders | `aligner="argmax" | "itermax" | "mai"` |
| **AER** (Och & Ney) | 2003 | Standard evaluation harness for the word aligners themselves | `ryokai.eval.aer(...)` / `evaluate_aligner(...)` |

The role-label canonicalisation (`ryokai/data/labelconfig.yaml`) collapses PropBank / Chinese PropBank / AnCora / reference-and-continuation tags into one ontology so the same scorer works across languages.

## Install

```bash
pip install ryokai
```

The class is `Ryokai`; `MEANT` is kept as a backwards-compatible alias since most of the literature uses that name.

## Quickstart

```python
from ryokai import Ryokai

r = Ryokai(lang="en")

score = r.score(
    reference="The cat sat on the mat yesterday.",
    hypothesis="A cat was sitting on the mat yesterday.",
)
print(score)
# Score(precision=0.74, recall=0.82, f1=0.78,
#       n_frames_ref=1, n_frames_hyp=1, n_aligned_predicates=1)

float(score)   # 0.78  — convenient shortcut, returns f1
```

Corpus-level:

```python
scores = r.score_corpus(refs, hyps)
corpus_f1 = sum(s.f1 for s in scores) / len(scores)
```

Any of the 13 languages — same call, just change `lang`:

```python
Ryokai("zh").score("猫坐在垫子上。", "一只猫在垫子上。")
Ryokai("hi").score("बिल्ली चटाई पर बैठी।", "एक बिल्ली चटाई पर बैठी थी।")
```

## Variants

### Default — frame-based scoring (MEANT 2.0)

Predicate-argument frames + Hungarian matching + weighted role F-score.

```python
Ryokai(lang="en").score(reference, hypothesis)
```

### Reference-free (XMEANT / YiSi-2)

Score MT directly against the **source** sentence — no reference needed. Works because the embedding backbone is multilingual.

```python
r = Ryokai(lang="zh")                            # MT output language
r.score_xlingual(
    source="The cat sat on the mat yesterday.",
    hypothesis="猫昨天坐在垫子上。",
    source_lang="en",
)
```

### No-SRL fallback (YiSi-1 / WOLVESAAR / SimAlign)

Skip SRL entirely and score by word alignment + embedding similarity. Five aligners are available, trading off speed for faithfulness to the literature:

| `aligner=` | Backbone used | Matching | Notes |
| ---------- | ------------- | -------- | ----- |
| `"sentence"` *(default)* | sentence encoder (MiniLM) | Hungarian | Fast, no extra download. Tokens embedded out of context — coarse approximation. |
| `"hungarian"` | contextual token encoder (XLM-R / Qwen3 / Jina / …) | scipy 1-to-1 Hungarian | Sultan et al. (2014) style: contextual vectors + exact-match prior + threshold. |
| `"argmax"`   | contextual token encoder | bidirectional argmax intersection | SimAlign Inter: each src picks its best tgt; keep only mutual best. Allows many-to-many. |
| `"itermax"`  | contextual token encoder | argmax intersection + iterative growth | SimAlign IterMax — **recommended for quality**. Adds back unaligned words above the threshold. |
| `"mai"`      | contextual token encoder | union of `hungarian` ∪ `argmax` ∪ `itermax` | SimAlign `mai` combinator — highest recall, useful as upper bound. |

```python
# Fast — uses the already-loaded MiniLM, no extra model download
Ryokai(lang="en", use_srl=False).score(ref, hyp)

# Sultan et al.: contextual encoder + threshold + exact-match prior
Ryokai(lang="en", use_srl=False, aligner="hungarian").score(ref, hyp)

# SimAlign IterMax — best F1 in the SimAlign paper
Ryokai(lang="en", use_srl=False, aligner="itermax",
       content_only=True, threshold=0.5).score(ref, hyp)
```

Extra flags for all no-SRL modes:

- `content_only=True` — drop stopwords + punctuation before alignment (Sultan et al. / WOLVESAAR `prop_harmonic` feature). Per-language stopwords bundled for all 13 languages in `ryokai/data/stopwords.yaml`.
- `aggregation="harmonic"` — be explicit about the WOLVESAAR harmonic mean of per-sentence aligned-content-word proportions (mathematically the same as F1, the name is for clarity).
- `threshold=0.5` — drop alignments with cosine below this (contextual aligners only).
- `exact_match_shortcut=True` — case-insensitive surface equality boosts a pair's similarity to 1.0 (contextual aligners only). Sultan et al.'s "lexical prior" cascade in modern dress.

### No-GPU static backend

For a no-forward-pass alternative (fastText-style):

```python
from ryokai import StaticEmbeddingSimBackend
from ryokai.scorer import score_nosrl

# uses only XLM-R's input embedding layer — no transformer forward pass
score_nosrl(ref, hyp, aligner="itermax",
            sim_backend=StaticEmbeddingSimBackend("xlm-r-base"))

# or actual fastText vectors via gensim (optional install)
StaticEmbeddingSimBackend(fasttext_path="cc.en.300.vec")
```

## Embedding backbones

Both backends — `EmbeddingSimBackend` for sentence/argument similarity and `ContextualTokenSimBackend` for word alignment — accept any HuggingFace model id, plus a set of shorthand presets:

```python
from ryokai import Ryokai, EmbeddingSimBackend

# Upgrade the sentence-level backbone (predicate / argument similarity)
Ryokai(
    lang="en",
    sim_backend=EmbeddingSimBackend("qwen3-0.6b"),       # or "jina-v3", "nemotron-8b", or any HF id
)
```

For the contextual word aligner backbone (used by `aligner="hungarian"/"argmax"/"itermax"/"mai"`), instantiate it explicitly and pass it to `score_nosrl`:

```python
from ryokai import ContextualTokenSimBackend
from ryokai.scorer import score_nosrl

backend = ContextualTokenSimBackend("qwen3-0.6b", layer=8)
# score_nosrl currently instantiates a default contextual backend
# internally — to plug in your custom one, drive the aligner directly:
pairs, ref_words, hyp_words = backend.align(ref, hyp, method="itermax")
```

### Preset catalogue

| Preset key            | Model                                                       | Year     | Backend |
| --------------------- | ----------------------------------------------------------- | -------- | ------- |
| `minilm`              | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 2021     | sentence (default) |
| `mpnet`               | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 2021     | sentence |
| `mbert`               | bert-base-multilingual-cased                                | 2019     | contextual |
| `xlm-r-base`          | xlm-roberta-base                                            | 2020     | contextual (default) |
| `xlm-r-large`         | xlm-roberta-large                                           | 2020     | contextual |
| `mdeberta-v3`         | microsoft/mdeberta-v3-base                                  | 2022     | contextual |
| `me5-large`           | intfloat/multilingual-e5-large                              | 2024     | sentence |
| `me5-large-instruct`  | intfloat/multilingual-e5-large-instruct                     | 2024     | sentence |
| `bge-m3`              | BAAI/bge-m3                                                 | 2024     | sentence + contextual |
| `mxbai-large`         | mixedbread-ai/mxbai-embed-large-v1                          | 2024     | sentence |
| `snowflake-arctic-v2` | Snowflake/snowflake-arctic-embed-l-v2.0                     | 2024     | sentence + contextual |
| `nomic-v2`            | nomic-ai/nomic-embed-text-v2-moe                            | 2025     | sentence |
| `jina-v2-base`        | jinaai/jina-embeddings-v2-base-en                           | 2023     | sentence + contextual |
| `jina-v3`             | jinaai/jina-embeddings-v3                                   | 2024     | sentence + contextual |
| `embedding-gemma`     | google/embeddinggemma-300m                                  | 2025     | sentence + contextual |
| **`qwen3-0.6b`**      | Qwen/Qwen3-Embedding-0.6B                                   | **2025** | sentence + contextual |
| `qwen3-4b`            | Qwen/Qwen3-Embedding-4B                                     | 2025     | sentence + contextual |
| `qwen3-8b`            | Qwen/Qwen3-Embedding-8B                                     | 2025     | sentence + contextual |
| **`nemotron-8b`**     | nvidia/llama-embed-nemotron-8b                              | **2025** *(#1 MMTEB Oct 2025)* | sentence + contextual |

Any non-preset string is passed straight through to `transformers.AutoModel` / `sentence_transformers.SentenceTransformer` (with `trust_remote_code=True` so models like Jina v3 / Nomic v2 that ship custom code work out of the box).

### SimAlign-inspired extras on the contextual backend

```python
from ryokai import ContextualTokenSimBackend

# pick a middle layer (SimAlign found layer 8 of mBERT-base / XLM-R-base
# beats the last layer for word alignment)
ContextualTokenSimBackend("xlm-r-base", layer=8)

# drop alignments where word positions differ by more than 30% of sentence length
ContextualTokenSimBackend("qwen3-0.6b", max_distortion=0.3)
```

These plug into any of the `aligner="hungarian"/"argmax"/"itermax"/"mai"` modes via the backend you pass in.

## Evaluating the aligners themselves (AER)

For benchmarking the word aligners against a gold word-alignment dataset (WPT, Hansards, Europarl, …), `ryokai.eval` exposes the standard Och & Ney (2003) AER metric and helpers:

```python
from ryokai import ContextualTokenSimBackend
from ryokai.eval import aer, evaluate_aligner

# Single-pair AER
predicted = [(0, 0, 0.9), (1, 1, 0.85)]   # (i, j, similarity) tuples are accepted
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

## CLI

```bash
ryokai --ref ref.txt --hyp hyp.txt --lang en
# ryokai = 0.71  (n=2000)

ryokai --ref ref.txt --hyp hyp.txt --lang en --no-corpus-avg > per_sentence.f1
```

## Architecture

```
ryokai/
├── __init__.py             # Ryokai facade (MEANT alias) + score_xlingual()
├── _langs.py               # 13-language registry
├── labelconfig.py          # load + apply SRL role aliasing
├── graph.py                # SRLGraph, Frame, Argument dataclasses
├── match.py                # scipy.linear_sum_assignment wrapper
├── scorer.py               # Score dataclass + frame-based + no-SRL scorers
├── stopwords.py            # per-language stopword loader (content-word filter)
├── eval.py                 # AER / precision / recall / F1 / evaluate_aligner
├── cli.py                  # argparse entry point (`ryokai` script)
├── data/
│   ├── labelconfig.yaml    # ported from meant.labelconfig (PropBank/AnCora/...)
│   └── stopwords.yaml      # minimal closed-class function-word lists for 13 langs
├── srl/
│   ├── __init__.py         # SRLBackend ABC
│   ├── hf_pos.py           # default: multilingual XLM-R UDPOS heuristic SRL
│   └── hf_backend.py       # plug-in: any HF token-classification SRL model
└── sim/
    ├── __init__.py
    ├── embeddings.py       # sentence-level: EmbeddingSimBackend + presets
    ├── contextual.py       # contextual word vectors + ArgMax/IterMax/MAI/Hungarian/distortion/layer
    └── static.py           # embedding-layer-only (no forward pass) + optional fastText
```

The **default SRL backend** (`HFPOSHeuristicSRLBackend`) is intentionally shallow — it tags every token via a single multilingual XLM-R UDPOS model and treats consecutive `VERB`/`AUX` spans as predicates with the nearest noun phrases on each side becoming `ARG0`/`ARG1`/`ARG2`. For MT eval what matters is **consistent treatment of ref and hyp**, not absolute SRL accuracy.

For PropBank-grade SRL on a specific language, plug in a real model:

```python
from ryokai import Ryokai, HFTokenClassifierSRLBackend

Ryokai(
    lang="en",
    srl_backend=HFTokenClassifierSRLBackend("liaad/srl-en_xlmr-base"),
).score(ref, hyp)
```

## Custom role weights

`Ryokai(lang=…, weights={...})` accepts any `{role_class: float}` dict. Defaults give equal weight to V/A0/A1, 0.8 to A2/NEG, 0.6 to TMP/LOC, less to modifiers. Setting a weight to 0 ignores that role class entirely.

## Running the tests

Fast tests (label aliasing, scorer arithmetic, matching, presets, AER, stopwords) — no model downloads:

```bash
pytest -m "not slow"     # 66 tests, < 5s
```

Full end-to-end with model downloads (~1.2 GB on first run, then cached):

```bash
pytest -m slow           # 62 tests across all 13 languages × every aligner
```

## Citations

If you use ryokai in research, please cite the original technique papers (we glue them together; the credit belongs to their authors):

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

@article{sultan-etal-2014-back,
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

@article{och-ney-2003-systematic,
  title   = {A Systematic Comparison of Various Statistical Alignment Models},
  author  = {Och, Franz Josef and Ney, Hermann},
  journal = {Computational Linguistics},
  year    = {2003},
}
```

## License

MIT — see `LICENSE`.
