# pymeant

Python port of the **MEANT 2.0** machine-translation evaluation metric, with **XMEANT** (reference-free) and **YiSi-1** (no-SRL) modes built in.

Pure PyTorch + HuggingFace `transformers` — no Stanza, no spaCy, no external parsers. Two HF models cover **all 13 MEANT languages** (`en`, `de`, `fr`, `es`, `cs`, `fi`, `hi`, `lv`, `pl`, `ro`, `ru`, `tr`, `zh`) in a single install:

- POS / shallow SRL: [`wietsedv/xlm-roberta-base-ft-udpos28`](https://huggingface.co/wietsedv/xlm-roberta-base-ft-udpos28) — ~1.1 GB once.
- Multilingual embeddings: [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — ~110 MB once, used for both **same-language** and **cross-language** similarity.

## Why MEANT?

Unlike BLEU/chrF (n-gram overlap) or COMET/BERTScore (sentence-level embeddings), MEANT scores translations by **semantic-role agreement**:

1. Parse the reference and hypothesis into predicate-argument frames.
2. Hungarian-match predicates by lexical similarity.
3. For each matched predicate, Hungarian-match the arguments under the same canonical role (`A0`, `A1`, `TMP`, `LOC`, …).
4. Aggregate to a weighted F-score over role labels.

The role-label canonicalisation (`pymeant/data/labelconfig.yaml`) collapses PropBank / Chinese PropBank / AnCora / reference-and-continuation tags into one ontology, so the same scorer works across languages.

## Install

```bash
pip install pymeant
```

## Quickstart

```python
from pymeant import MEANT

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

### No-SRL fallback (YiSi-1 style)

Skip SRL entirely — token-level Hungarian alignment with embedding cosine. Faster and downloads fewer models; comparable to YiSi-1.

```python
MEANT(lang="en", use_srl=False).score(reference, hypothesis)
```

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
pymeant --ref ref.txt --hyp hyp.txt --lang en
# MEANT = 0.71  (n=2000)

pymeant --ref ref.txt --hyp hyp.txt --lang en --no-corpus-avg > per_sentence.f1
```

## Architecture

```
pymeant/
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
from pymeant import MEANT, HFTokenClassifierSRLBackend

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

If you use pymeant in research, please cite the original metric papers alongside this port:

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

## License

MIT — see `LICENSE`.
