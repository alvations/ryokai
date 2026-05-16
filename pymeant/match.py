"""Max-weighted bipartite matching via scipy's Hungarian algorithm.

MEANT uses this twice: once to align predicates between the reference and
MT output, once (per matched predicate pair) to align their arguments.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def max_match(sim: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Given a similarity matrix shape (n, m), return the matching that
    maximises total similarity.

    Returns ([(i, j), ...], total_similarity).
    """
    if sim.size == 0:
        return [], 0.0
    # linear_sum_assignment minimises cost -> negate similarity
    rows, cols = linear_sum_assignment(-sim)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    total = float(sim[rows, cols].sum())
    return pairs, total
