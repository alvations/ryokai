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

r = Ryokai(lang="en")
score = r.score(
    reference="The cat sat on the mat yesterday.",
    hypothesis="A cat was sitting on the mat yesterday.",
)
print(score)
# Score(precision=0.74, recall=0.82, f1=0.78, ...)

float(score)   # 0.78  — convenient shortcut, returns f1
```

## Variants

```python
# 1. Frame-based MEANT 2.0 (default)
Ryokai(lang).score(ref, hyp)

# 2. Reference-free XMEANT / YiSi-2 — score MT vs source, no reference
Ryokai(lang).score_xlingual(src, hyp, source_lang)

# 3. No-SRL YiSi-1 / WOLVESAAR / SimAlign — word alignment + embedding F-score
Ryokai(lang, use_srl=False, aligner="itermax").score(ref, hyp)
```

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for flags, aligner choices, embedding-backbone swaps, AER evaluation harness, CLI, architecture, and custom role weights.

## References

Ryokai is glue around several published techniques. Credit belongs to their authors. The table groups them by family.

### Semantic-frame MT evaluation

| Technique | Citation | Link |
| --------- | -------- | ---- |
| **MEANT** | Lo & Wu (2011) — *MEANT: An inexpensive, high-accuracy, semi-automatic metric for evaluating translation utility based on semantic roles*, ACL 2011 | [P11-2042](https://aclanthology.org/P11-2042/) |
| **XMEANT** | Lo, Beloucif, Saers, Wu (2014) — *XMEANT: Better semantic MT evaluation without reference translations*, ACL 2014 (Short Papers), pp. 765–771 | [P14-2124](https://aclanthology.org/P14-2124/) |
| **MEANT 2.0** | Lo (2017) — *MEANT 2.0: Accurate semantic MT evaluation for any output language*, WMT 2017 | [W17-4767](https://aclanthology.org/W17-4767/) |

### Embedding-based MT evaluation

| Technique | Citation | Link |
| --------- | -------- | ---- |
| **Doc-embedding adequacy** | Vela & Tan (2015) — *Predicting Machine Translation Adequacy with Document Embeddings*, WMT 2015, pp. 402–410 | [W15-3051](https://aclanthology.org/W15-3051/) |
| **WOLVESAAR** | Bechara, Gupta, Tan, Orăsan, Mitkov, van Genabith (2016) — *WOLVESAAR at SemEval-2016 Task 1: Replicating the Success of Monolingual Word Alignment and Neural Embeddings for STS*, SemEval-2016, pp. 634–639 | [S16-1096](https://aclanthology.org/S16-1096/) |
| **YiSi** | Lo (2019) — *YiSi — a Unified Semantic MT Quality Evaluation and Estimation Metric for Languages with Different Levels of Available Resources*, WMT 2019 | [W19-5358](https://aclanthology.org/W19-5358/) |

### Word alignment

| Technique | Citation | Link |
| --------- | -------- | ---- |
| **Monolingual aligner** | Sultan, Bethard, Sumner (2014) — *Back to Basics for Monolingual Alignment: Exploiting Word Similarity and Contextual Evidence*, TACL 2014 | [Q14-1018](https://aclanthology.org/Q14-1018/) |
| **SimAlign** | Jalili Sabet, Dufter, Yvon, Schütze (2020) — *SimAlign: High Quality Word Alignments without Parallel Training Data using Static and Contextualized Embeddings*, Findings of EMNLP 2020 | [2020.findings-emnlp.147](https://aclanthology.org/2020.findings-emnlp.147/) |

### Alignment evaluation

| Technique | Citation | Link |
| --------- | -------- | ---- |
| **AER** | Och & Ney (2003) — *A Systematic Comparison of Various Statistical Alignment Models*, Computational Linguistics 29(1), pp. 19–51 | [J03-1002](https://aclanthology.org/J03-1002/) |

## License

MIT — see [`LICENSE`](LICENSE).
