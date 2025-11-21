# src/core_sp/sp_metrics.py

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, Literal, Optional, Dict
from dataclasses import dataclass


LayoutType = Literal["grid", "line", "circle", "cluster", "random"]


@dataclass
class SPComponents:
    """Individual components of Structural Preservation."""
    sp_adj: float
    sp_ord: float
    sp_clu: float

    @property
    def total(self) -> float:
        """Composite SP score in [0, 1]."""
        return float((self.sp_adj + self.sp_ord + self.sp_clu) / 3.0)


def compute_knn_graph(coords: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Compute a symmetric k-NN adjacency matrix.

    Parameters
    ----------
    coords : (N, d) array
        Coordinates of N items in d-dimensional space.
    k : int
        Number of nearest neighbours per node (excluding self).

    Returns
    -------
    adj : (N, N) bool array
        Symmetric adjacency matrix; adj[i, j] = True if i and j are neighbours.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if k >= n:
        raise ValueError(f"k must be < N; got k={k}, N={n}")

    dists = cdist(coords, coords, metric="euclidean")
    np.fill_diagonal(dists, np.inf)

    # argsort returns sorted indices along axis
    knn_idx = np.argsort(dists, axis=1)[:, :k]

    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in knn_idx[i]:
            adj[i, j] = True
            adj[j, i] = True  # make symmetric

    return adj


def jaccard_similarity_bool(a: np.ndarray, b: np.ndarray) -> float:
    """
    Jaccard similarity between two boolean adjacency matrices.

    Parameters
    ----------
    a, b : (N, N) bool arrays

    Returns
    -------
    float in [0, 1]
    """
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # degenerate case: no edges in either graph
    return float(inter / union)


def compute_sp_adj(
    base_coords: np.ndarray,
    trans_coords: np.ndarray,
    k: int = 4,
) -> float:
    """
    Structural preservation based on adjacency (SP_adj).

    Parameters
    ----------
    base_coords : (N, d) array
        Baseline coordinates.
    trans_coords : (N, d) array
        Transformed coordinates.
    k : int
        Neighbourhood size.

    Returns
    -------
    float
        Jaccard similarity between k-NN graphs in [0, 1].
    """
    adj_base = compute_knn_graph(base_coords, k=k)
    adj_trans = compute_knn_graph(trans_coords, k=k)
    return jaccard_similarity_bool(adj_base, adj_trans)


def compute_sp_ord(
    base_coords: np.ndarray,
    trans_coords: np.ndarray,
    layout_type: LayoutType,
) -> float:
    """
    Structural preservation based on order (SP_ord).

    For layouts with an intrinsic 1D order (grid, line, cluster),
    we compute Kendall's tau between baseline order indices and
    the order induced by transformed coordinates.

    Parameters
    ----------
    base_coords : (N, d) array
    trans_coords : (N, d) array
    layout_type : {"grid", "line", "circle", "cluster", "random"}

    Returns
    -------
    float
        SP_ord in [0, 1].
    """
    from scipy.stats import kendalltau

    n = base_coords.shape[0]
    base_order = np.arange(n)

    # For now: use x-axis as ordering axis (refine per layout_type if necessary)
    trans_order = np.argsort(trans_coords[:, 0])

    if layout_type == "random":
        return 1.0  # no intrinsic order to preserve

    tau, _ = kendalltau(base_order, trans_order)
    if np.isnan(tau):
        return 0.0
    # map [-1, 1] → [0, 1]
    return float((tau + 1.0) / 2.0)


def compute_sp_clu(
    base_coords: np.ndarray,
    trans_coords: np.ndarray,
    layout_type: LayoutType,
    n_clusters: int = 4,
    random_state: int = 0,
) -> float:
    """
    Structural preservation based on clustering (SP_clu).

    Parameters
    ----------
    base_coords : (N, d) array
    trans_coords : (N, d) array
    layout_type : LayoutType
        Only "cluster" is treated as having meaningful cluster structure.
    n_clusters : int
        Number of clusters to use when inferring labels.
    random_state : int
        Seed for clustering algorithm.

    Returns
    -------
    float
        SP_clu in [0, 1].
    """
    if layout_type != "cluster":
        return 1.0

    from sklearn.cluster import KMeans

    base = np.asarray(base_coords, dtype=float)
    trans = np.asarray(trans_coords, dtype=float)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    base_labels = km.fit_predict(base)
    trans_labels = km.fit_predict(trans)

    # best alignment between label permutations could be computed;
    # for a first pass we assume clusters are relatively stable.
    matches = (base_labels == trans_labels).sum()
    return float(matches / base.shape[0])


def compute_sp_components(
    base_coords: np.ndarray,
    trans_coords: np.ndarray,
    layout_type: LayoutType,
    k: int = 4,
    n_clusters: int = 4,
    random_state: int = 0,
) -> SPComponents:
    """
    Compute all SP components for a pair of layouts.

    Returns
    -------
    SPComponents
    """
    sp_adj = compute_sp_adj(base_coords, trans_coords, k=k)
    sp_ord = compute_sp_ord(base_coords, trans_coords, layout_type=layout_type)
    sp_clu = compute_sp_clu(
        base_coords, trans_coords, layout_type=layout_type,
        n_clusters=n_clusters, random_state=random_state,
    )
    return SPComponents(sp_adj=sp_adj, sp_ord=sp_ord, sp_clu=sp_clu)


def compute_sp_total(
    base_coords: np.ndarray,
    trans_coords: np.ndarray,
    layout_type: LayoutType,
    k: int = 4,
    n_clusters: int = 4,
    random_state: int = 0,
) -> float:
    """
    Convenience wrapper: directly return composite SP.

    Parameters
    ----------
    base_coords, trans_coords : (N, d)
    layout_type : LayoutType
    k : int
    n_clusters : int
    random_state : int

    Returns
    -------
    float
        Composite SP in [0, 1].
    """
    comps = compute_sp_components(
        base_coords=base_coords,
        trans_coords=trans_coords,
        layout_type=layout_type,
        k=k,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    return comps.total
