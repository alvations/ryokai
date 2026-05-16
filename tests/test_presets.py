"""Preset registries for sentence + contextual embedding backbones."""
from ryokai.sim.contextual import CTX_PRESETS, ContextualTokenSimBackend
from ryokai.sim.embeddings import EMBEDDING_PRESETS, EmbeddingSimBackend


def test_embedding_presets_include_modern_2025_models():
    expected = {
        "minilm", "mpnet", "me5-large", "bge-m3",
        "qwen3-0.6b", "qwen3-4b", "qwen3-8b",
        "jina-v3", "nemotron-8b", "snowflake-arctic-v2",
        "embedding-gemma", "nomic-v2",
    }
    assert expected.issubset(set(EMBEDDING_PRESETS))


def test_contextual_presets_include_modern_2025_models():
    expected = {
        "mbert", "xlm-r-base", "xlm-r-large", "mdeberta-v3",
        "qwen3-0.6b", "qwen3-4b", "qwen3-8b",
        "jina-v3", "bge-m3", "nemotron-8b", "embedding-gemma",
    }
    assert expected.issubset(set(CTX_PRESETS))


def test_embedding_backend_resolves_preset_alias():
    be = EmbeddingSimBackend("qwen3-0.6b")
    assert be.model_id == "Qwen/Qwen3-Embedding-0.6B"


def test_embedding_backend_accepts_raw_hf_id():
    be = EmbeddingSimBackend("BAAI/bge-m3")
    assert be.model_id == "BAAI/bge-m3"


def test_contextual_backend_resolves_preset_alias():
    be = ContextualTokenSimBackend("jina-v3")
    assert be.model_id == "jinaai/jina-embeddings-v3"


def test_contextual_backend_accepts_raw_hf_id():
    be = ContextualTokenSimBackend("microsoft/mdeberta-v3-base")
    assert be.model_id == "microsoft/mdeberta-v3-base"
