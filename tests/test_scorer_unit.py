"""Scorer tests with hand-crafted SRLGraphs — no SRL / embedding download."""
import numpy as np

from pymeant.graph import Argument, Frame, SRLGraph
from pymeant.scorer import DEFAULT_WEIGHTS, score_pair


class _IdentitySim:
    """Sim backend that returns 1.0 iff strings match exactly, else 0.0.
    Lets us write deterministic scorer tests without loading embeddings."""

    def sim(self, a, b):
        out = np.zeros((len(a), len(b)), dtype=np.float32)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x == y:
                    out[i, j] = 1.0
        return out


def _make_graph(sentence, frames):
    g = SRLGraph(sentence=sentence)
    for pred, args in frames:
        f = Frame(predicate=pred)
        for label, text in args:
            f.arguments.append(Argument(label=label, raw_label=label, text=text))
        g.frames.append(f)
    return g


def test_identical_graphs_score_one():
    g = _make_graph("the cat sat", [("sit", [("A0", "the cat")])])
    score = score_pair(g, g, sim_backend=_IdentitySim())
    assert score.f1 == 1.0
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.n_aligned_predicates == 1


def test_completely_disjoint_graphs_score_zero():
    g1 = _make_graph("eat", [("eat", [("A0", "alice"), ("A1", "cake")])])
    g2 = _make_graph("sleep", [("sleep", [("A0", "bob"), ("A1", "bed")])])
    score = score_pair(g1, g2, sim_backend=_IdentitySim())
    assert score.f1 == 0.0


def test_partial_overlap_is_between_0_and_1():
    g_ref = _make_graph("eat", [("eat", [("A0", "alice"), ("A1", "cake")])])
    g_hyp = _make_graph("eat", [("eat", [("A0", "alice"), ("A1", "pie")])])
    score = score_pair(g_ref, g_hyp, sim_backend=_IdentitySim())
    assert 0.0 < score.f1 < 1.0


def test_empty_ref_and_hyp_is_perfect():
    g = SRLGraph(sentence="x")
    score = score_pair(g, g, sim_backend=_IdentitySim())
    assert score.f1 == 1.0


def test_one_empty_one_nonempty_is_zero():
    g_empty = SRLGraph(sentence="x")
    g_full = _make_graph("y", [("v", [("A0", "a")])])
    score = score_pair(g_empty, g_full, sim_backend=_IdentitySim())
    assert score.f1 == 0.0


def test_extra_unaligned_hyp_predicate_drops_precision():
    g_ref = _make_graph("x", [("eat", [("A0", "a"), ("A1", "b")])])
    g_hyp = _make_graph(
        "x",
        [("eat", [("A0", "a"), ("A1", "b")]),
         ("run", [("A0", "c")])],
    )
    score = score_pair(g_ref, g_hyp, sim_backend=_IdentitySim())
    assert score.recall == 1.0
    assert score.precision < 1.0
    assert score.f1 < 1.0


def test_weights_dict_is_applied():
    g_ref = _make_graph("x", [("eat", [("A0", "a"), ("TMP", "yesterday")])])
    g_hyp = _make_graph("x", [("eat", [("A0", "a"), ("TMP", "today")])])
    # baseline: A0 matches, TMP doesn't
    s_baseline = score_pair(g_ref, g_hyp, sim_backend=_IdentitySim())
    # zero out TMP's weight -> A0 alone fully matches -> score goes up
    weights = dict(DEFAULT_WEIGHTS)
    weights["TMP"] = 0.0
    s_no_tmp = score_pair(g_ref, g_hyp, sim_backend=_IdentitySim(), weights=weights)
    assert s_no_tmp.f1 > s_baseline.f1
