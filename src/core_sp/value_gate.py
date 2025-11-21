"""Value gate mechanism for O-4 experiments.

This module provides the value-gated coupling mechanism that increases
SSC by modulating spatial arrangement based on semantic similarity.

The implementation is adapted from exp_13_value_gate_sweep.py with
generalized interface for v2 experiments.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def apply_value_gate(
    base_coords: np.ndarray,
    embeddings: np.ndarray,
    lam: float,
    seed: int = 42,
    radius: float = 1.0,
    layout: str = 'circle'
) -> np.ndarray:
    """
    Apply value gate parameter λ to arrange items spatially.
    
    λ=0: Pure random arrangement (no value pressure)
    λ=1: Perfect semantic-spatial alignment (max value pressure)
    
    Parameters
    ----------
    base_coords : ndarray, shape (n_items, 2) or (n_items, 3)
        Base coordinates (used for reference, not modified)
    embeddings : ndarray, shape (n_items, dim)
        Semantic embeddings
    lam : float in [0, 1]
        Value gate parameter
    seed : int, default=42
        Random seed for reproducibility
    radius : float, default=1.0
        Layout radius (for circle layout)
    layout : str, default='circle'
        Layout type ('circle' supported in v2.0.0)
    
    Returns
    -------
    coords_gated : ndarray, shape (n_items, 2) or (n_items, 3)
        Spatially arranged coordinates under value gate
    
    Notes
    -----
    This function implements the value-gated coupling mechanism validated
    in O-4. The algorithm:
    1. Computes semantic distance matrix D_sem
    2. Generates random distance matrix D_rand
    3. Combines: D = (1-λ)*D_rand + λ*D_sem
    4. Applies greedy TSP-like ordering to minimize combined distance
    5. Places ordered items on circle layout
    
    References
    ----------
    Adapted from exp_13_value_gate_sweep.py (v1.1.2)
    
    Examples
    --------
    >>> from src.core_sp.generators import generate_semantic_embeddings
    >>> rng = np.random.default_rng(42)
    >>> embeddings = generate_semantic_embeddings(20, 100, rng)
    >>> base_coords = np.random.uniform(-1, 1, (20, 2))
    >>> coords_gated = apply_value_gate(base_coords, embeddings, lam=0.5, seed=42)
    >>> coords_gated.shape
    (20, 2)
    """
    rng = np.random.default_rng(seed)
    n_items = embeddings.shape[0]
    
    # Semantic distances
    D_sem = squareform(pdist(embeddings, metric='correlation'))
    
    # Random distances
    D_rand = rng.uniform(0, 1, (n_items, n_items))
    D_rand = (D_rand + D_rand.T) / 2  # Symmetrize
    np.fill_diagonal(D_rand, 0)
    
    # Combine: D = (1-λ)*D_rand + λ*D_sem
    D_combined = (1 - lam) * D_rand + lam * D_sem
    
    # Greedy TSP-like ordering to minimize combined distance
    start = rng.integers(0, n_items)
    ordering = [start]
    remaining = set(range(n_items)) - {start}
    current = start
    
    while remaining:
        distances = [(D_combined[current, node], node) for node in remaining]
        _, nearest = min(distances)
        ordering.append(nearest)
        remaining.remove(nearest)
        current = nearest
    
    ordering = np.array(ordering)
    
    # Generate layout coordinates
    if layout == 'circle':
        angles = 2 * np.pi * np.arange(n_items) / n_items
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        coords = np.column_stack([x, y])
    else:
        raise NotImplementedError(f"Layout '{layout}' not supported in v2.0.0")
    
    # Apply ordering
    ordered_coords = np.zeros_like(coords)
    for i, item_idx in enumerate(ordering):
        ordered_coords[item_idx] = coords[i]
    
    return ordered_coords
