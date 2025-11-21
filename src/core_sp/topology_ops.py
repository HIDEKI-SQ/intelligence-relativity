# src/core_sp/topology_ops.py

from __future__ import annotations

import numpy as np
from typing import Tuple


def edge_rewire(
    adj: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly rewire a proportion p of edges in an undirected adjacency matrix.

    Parameters
    ----------
    adj : (N, N) bool array
        Symmetric adjacency matrix.
    p : float in [0, 1]
        Proportion of edges to rewire.
    rng : np.random.Generator

    Returns
    -------
    adj_new : (N, N) bool array
    """
    adj = adj.astype(bool)
    n = adj.shape[0]
    assert adj.shape == (n, n)

    # work on upper triangle to avoid double counting
    tri_idx = np.triu_indices(n, k=1)
    edges = np.where(adj[tri_idx])[0]

    if edges.size == 0 or p <= 0.0:
        return adj.copy()

    n_edges = edges.size
    n_rewire = max(1, int(round(p * n_edges)))

    # choose edges to remove
    remove_idx = rng.choice(edges, size=n_rewire, replace=False)
    adj_new = adj.copy()

    # remove selected edges
    rows = tri_idx[0][remove_idx]
    cols = tri_idx[1][remove_idx]
    adj_new[rows, cols] = False
    adj_new[cols, rows] = False

    # add same number of new random edges
    # naive approach; could be refined to avoid repeated attempts
    added = 0
    attempts = 0
    max_attempts = n_rewire * 10

    while added < n_rewire and attempts < max_attempts:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            attempts += 1
            continue
        if adj_new[i, j]:
            attempts += 1
            continue
        adj_new[i, j] = True
        adj_new[j, i] = True
        added += 1
        attempts += 1

    return adj_new


def permute_coords(
    coords: np.ndarray,
    rng: np.random.Generator,
    p: float,
) -> np.ndarray:
    """
    Randomly permute a proportion p of coordinates (order disruption).

    Parameters
    ----------
    coords : (N, d) array
    rng : np.random.Generator
    p : float in [0, 1]

    Returns
    -------
    coords_new : (N, d) array
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    coords_new = coords.copy()

    if p <= 0.0:
        return coords_new

    n_swap = max(1, int(round(p * n / 2)))
    for _ in range(n_swap):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        tmp = coords_new[i].copy()
        coords_new[i] = coords_new[j]
        coords_new[j] = tmp

    return coords_new


def random_relayout(
    coords: np.ndarray,
    rng: np.random.Generator,
    bounds: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    Completely randomize coordinates within given bounds
    (full destruction benchmark).

    Parameters
    ----------
    coords : (N, d) array
    rng : np.random.Generator
    bounds : (low, high)

    Returns
    -------
    coords_new : (N, d) array
    """
    coords = np.asarray(coords, dtype=float)
    n, d = coords.shape
    low, high = bounds
    coords_new = rng.uniform(low=low, high=high, size=(n, d))
    return coords_new
