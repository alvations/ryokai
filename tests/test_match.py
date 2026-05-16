"""Pure-Python tests for the Hungarian matching wrapper."""
import numpy as np

from ryokai.match import max_match


def test_identity_matrix_one_to_one_diagonal():
    sim = np.eye(3, dtype=np.float32)
    pairs, total = max_match(sim)
    assert sorted(pairs) == [(0, 0), (1, 1), (2, 2)]
    assert total == 3.0


def test_max_match_picks_higher_sum():
    sim = np.array(
        [
            [0.1, 0.9],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    pairs, total = max_match(sim)
    # best assignment: (0,1)+(1,0) = 0.9+0.8 = 1.7
    assert sorted(pairs) == [(0, 1), (1, 0)]
    assert abs(total - 1.7) < 1e-5


def test_rectangular_matrix_more_cols():
    sim = np.array(
        [
            [0.2, 0.9, 0.5],
            [0.7, 0.1, 0.4],
        ],
        dtype=np.float32,
    )
    pairs, total = max_match(sim)
    assert len(pairs) == 2  # min(rows, cols)
    assert total > 0


def test_empty_matrix_returns_zero():
    sim = np.zeros((0, 5), dtype=np.float32)
    pairs, total = max_match(sim)
    assert pairs == []
    assert total == 0.0
