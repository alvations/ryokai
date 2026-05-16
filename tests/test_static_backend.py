"""Unit tests for the static-embedding backend (constructor / matcher dispatch).

The actual embedding lookup is exercised in the slow test suite — these
tests cover everything that doesn't require downloading models."""
import numpy as np
import pytest

from pymeant.sim.static import StaticEmbeddingSimBackend, _dispatch_match


def test_static_backend_resolves_ctx_preset_alias():
    be = StaticEmbeddingSimBackend("xlm-r-base")
    assert be.model_id == "xlm-roberta-base"


def test_static_backend_accepts_raw_hf_id():
    be = StaticEmbeddingSimBackend("BAAI/bge-m3")
    assert be.model_id == "BAAI/bge-m3"


def test_static_backend_distortion_default_none():
    be = StaticEmbeddingSimBackend()
    assert be.max_distortion is None


def test_static_backend_remembers_fasttext_path():
    be = StaticEmbeddingSimBackend(fasttext_path="/tmp/cc.en.300.vec")
    assert be.fasttext_path == "/tmp/cc.en.300.vec"


def test_dispatch_match_hungarian():
    sim = np.eye(3, dtype=np.float32)
    pairs = _dispatch_match(sim, "hungarian", threshold=0.5)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 0), (1, 1), (2, 2)]


def test_dispatch_match_argmax_intersection_only():
    # argmax intersection keeps only mutual best
    sim = np.array(
        [[0.9, 0.1, 0.05],
         [0.2, 0.8, 0.1],
         [0.1, 0.4, 0.6]],
        dtype=np.float32,
    )
    pairs = _dispatch_match(sim, "argmax", threshold=0.5)
    pairs_idx = sorted((i, j) for i, j, _ in pairs)
    assert pairs_idx == [(0, 0), (1, 1), (2, 2)]


def test_dispatch_match_itermax_picks_up_unaligned():
    # rectangular: 4 refs vs 3 hyps, argmax alone would orphan one ref
    sim = np.array(
        [[0.9, 0.1, 0.0],
         [0.1, 0.85, 0.05],
         [0.05, 0.05, 0.95],
         [0.6, 0.0, 0.7]],
        dtype=np.float32,
    )
    arg_pairs = _dispatch_match(sim, "argmax", threshold=0.5)
    iter_pairs = _dispatch_match(sim, "itermax", threshold=0.5)
    arg_refs = {i for i, _, _ in arg_pairs}
    iter_refs = {i for i, _, _ in iter_pairs}
    # itermax recovers at least as many refs as argmax
    assert iter_refs >= arg_refs


def test_dispatch_match_mai_is_superset_of_components():
    sim = np.array(
        [[0.9, 0.5, 0.1],
         [0.4, 0.85, 0.2],
         [0.1, 0.3, 0.95]],
        dtype=np.float32,
    )
    h = {(i, j) for i, j, _ in _dispatch_match(sim, "hungarian", 0.5)}
    a = {(i, j) for i, j, _ in _dispatch_match(sim, "argmax", 0.5)}
    it = {(i, j) for i, j, _ in _dispatch_match(sim, "itermax", 0.5)}
    mai = {(i, j) for i, j, _ in _dispatch_match(sim, "mai", 0.5)}
    assert mai == h | a | it


def test_dispatch_match_unknown_method_raises():
    with pytest.raises(ValueError):
        _dispatch_match(np.zeros((1, 1), dtype=np.float32), "bogus", 0.5)
