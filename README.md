# ryokai 了解

> *Ryokai* (了解, "understood / got it") — a unified Python library for **semantic machine-translation evaluation**, combining the strengths of MEANT 2.0, XMEANT, YiSi-1/2, WOLVESAAR, and SimAlign behind one clean API on top of modern multilingual embeddings.

Pure PyTorch + HuggingFace `transformers` — no Stanza, no spaCy, no external parsers. Two HF models cover all 13 supported languages (`en`, `de`, `fr`, `es`, `cs`, `fi`, `hi`, `lv`, `pl`, `ro`, `ru`, `tr`, `zh`) in a single install:

- POS / shallow SRL: [`wietsedv/xlm-roberta-base-ft-udpos28`](https://huggingface.co/wietsedv/xlm-roberta-base-ft-udpos28) — ~1.1 GB, downloaded once.
- Multilingual embeddings: [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — ~110 MB, used for both same-language and cross-language similarity.

Both are one-line swappable for any modern multilingual encoder (Qwen3-Embedding, Jina v3, BGE-M3, Nemotron-8B…) — see [Embedding backbones](DOCUMENTATION.md#embedding-backbones) in `DOCUMENTATION.md`.

## Install

```bash
pip install ryokai
```

## Quickstart

```python
from ryokai import Ryokai

scorer = Ryokai()
src_lang, tgt_lang = "en", "ja"

# Most common: reference-free, word alignment + embedding
# (XMEANT-lite / YiSi-2 / Doc-embedding adequacy cross-lingual)
scorer.score(source=src, hypothesis=hyp,
             source_lang=src_lang, target_lang=tgt_lang)
```

## Variants

One `.score()` call, four modes, dispatched by which arguments you pass. `srl=False` is the default — `ryokai` is no longer MEANT-first.

```python
from ryokai import Ryokai
scorer = Ryokai()
src_lang, tgt_lang = "en", "ja"

# Reference-free, word alignment + embedding (default, most common)
# E.g. Doc-embedding adequacy / YiSi-2 / XMEANT-lite
scorer.score(source=src, hypothesis=hyp,
             source_lang=src_lang, target_lang=tgt_lang)

# Reference-based, word alignment + embedding
# E.g. Doc-embedding adequacy / WOLVESAAR / YiSi-1 / SimAlign style
scorer.score(reference=ref, hypothesis=hyp, target_lang=tgt_lang)

# Reference-free, frame-based — XMEANT proper
scorer.score(source=src, hypothesis=hyp,
             source_lang=src_lang, target_lang=tgt_lang, srl=True)

# Reference-based, frame-based — MEANT 2.0
scorer.score(reference=ref, hypothesis=hyp, target_lang=tgt_lang, srl=True)
```

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for flags, aligner choices, embedding-backbone swaps, AER evaluation harness, CLI, architecture, and custom role weights.

## References

Ryokai is glue around several published techniques — credit belongs to their authors.

| Technique | Year | Citation | Category |
| --------- | ---- | -------- | -------- |
| MEANT                  | 2011 | Lo & Wu. [*MEANT: An inexpensive, high-accuracy, semi-automatic metric for evaluating translation utility based on semantic roles*](https://aclanthology.org/P11-2042/). ACL 2011. | Semantic-frame MT evaluation  |
| XMEANT                 | 2014 | Lo, Beloucif, Saers & Wu. [*XMEANT: Better semantic MT evaluation without reference translations*](https://aclanthology.org/P14-2124/). ACL 2014 (Short Papers). | Semantic-frame MT evaluation  |
| MEANT 2.0              | 2017 | Lo. [*MEANT 2.0: Accurate semantic MT evaluation for any output language*](https://aclanthology.org/W17-4767/). WMT 2017. | Semantic-frame MT evaluation  |
| Doc-embedding adequacy | 2015 | Vela & Tan. [*Predicting Machine Translation Adequacy with Document Embeddings*](https://aclanthology.org/W15-3051/). WMT 2015. | Embedding-based MT evaluation |
| WOLVESAAR              | 2016 | Bechara, Gupta, Tan, Orăsan, Mitkov & van Genabith. [*WOLVESAAR at SemEval-2016 Task 1: Replicating the Success of Monolingual Word Alignment and Neural Embeddings for Semantic Textual Similarity*](https://aclanthology.org/S16-1096/). SemEval-2016. | Embedding-based MT evaluation |
| YiSi                   | 2019 | Lo. [*YiSi — a Unified Semantic MT Quality Evaluation and Estimation Metric for Languages with Different Levels of Available Resources*](https://aclanthology.org/W19-5358/). WMT 2019. | Embedding-based MT evaluation |
| Monolingual aligner    | 2014 | Sultan, Bethard & Sumner. [*Back to Basics for Monolingual Alignment: Exploiting Word Similarity and Contextual Evidence*](https://aclanthology.org/Q14-1018/). TACL 2014. | Word alignment                |
| SimAlign               | 2020 | Jalili Sabet, Dufter, Yvon & Schütze. [*SimAlign: High Quality Word Alignments without Parallel Training Data using Static and Contextualized Embeddings*](https://aclanthology.org/2020.findings-emnlp.147/). Findings of EMNLP 2020. | Word alignment                |

## License

MIT — see [`LICENSE`](LICENSE).
