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

Ryokai is glue around several published techniques — credit belongs to their authors.

| Category | Technique | Author (year) |
| -------- | --------- | ------------- |
| Semantic-frame MT evaluation  | [MEANT](https://aclanthology.org/P11-2042/)                              | Lo & Wu (2011) |
| Semantic-frame MT evaluation  | [XMEANT](https://aclanthology.org/P14-2124/)                             | Lo, Beloucif, Saers & Wu (2014) |
| Semantic-frame MT evaluation  | [MEANT 2.0](https://aclanthology.org/W17-4767/)                          | Lo (2017) |
| Embedding-based MT evaluation | [Doc-embedding adequacy](https://aclanthology.org/W15-3051/)             | Vela & Tan (2015) |
| Embedding-based MT evaluation | [WOLVESAAR](https://aclanthology.org/S16-1096/)                          | Bechara, Gupta, Tan, Orăsan, Mitkov & van Genabith (2016) |
| Embedding-based MT evaluation | [YiSi](https://aclanthology.org/W19-5358/)                               | Lo (2019) |
| Word alignment                | [Monolingual aligner](https://aclanthology.org/Q14-1018/)                | Sultan, Bethard & Sumner (2014) |
| Word alignment                | [SimAlign](https://aclanthology.org/2020.findings-emnlp.147/)            | Jalili Sabet, Dufter, Yvon & Schütze (2020) |

## License

MIT — see [`LICENSE`](LICENSE).
