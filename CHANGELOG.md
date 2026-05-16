# Changelog

## 0.1.0 — initial release

Unified semantic MT-evaluation library (formerly internal name `pymeant`).

**Scoring modes**
- `MEANT(lang).score(ref, hyp)` — MEANT 2.0 semantic-frame F-score
  (Lo, WMT 2017) with the predicate-and-argument pipeline ported to
  pure Python.
- `MEANT(lang).score_xlingual(src, hyp, source_lang)` — XMEANT
  reference-free scoring (Lo et al. ACL 2014).
- `MEANT(lang, use_srl=False)` — YiSi-1 / WOLVESAAR style scoring
  without SRL.

**SRL backends**
- `HFPOSHeuristicSRLBackend` (default) — multilingual XLM-R UDPOS
  heuristic SRL via `wietsedv/xlm-roberta-base-ft-udpos28`.
- `HFTokenClassifierSRLBackend` — plug any HF token-classification
  SRL model.

**Word-alignment backends**
- `ContextualTokenSimBackend` — XLM-RoBERTa contextual word vectors
  with `layer` selection and `max_distortion` filter (SimAlign-style).
- `StaticEmbeddingSimBackend` — embedding-layer-only static vectors,
  optionally backed by fastText (.vec / .bin via gensim).
- Aligner choices: `sentence`, `hungarian`, `argmax`, `itermax`, `mai`.
- Sultan-style exact-match prior, content-word filtering, threshold,
  harmonic / F1 aggregation.

**Modern multilingual embedding presets**
- `EMBEDDING_PRESETS` and `CTX_PRESETS` cover MiniLM / mPNet / mE5 /
  BGE-M3 / mxbai-large / Snowflake Arctic v2 / Nomic v2 / Jina v2-v3 /
  EmbeddingGemma / Qwen3-Embedding (0.6B/4B/8B) / Llama-Embed-Nemotron-8B.

**Evaluation harness** (`ryokai.eval`)
- `aer(predicted, sure, possible)` — Och & Ney (2003) Alignment Error Rate.
- `precision` / `recall` / `f1` for word-alignment sets.
- `evaluate_aligner(backend, examples)` — corpus sweep.

**Tests**
- 66 fast unit tests (no model downloads).
- 62 slow tests across all 13 MEANT languages × aligners.
