"""Value gate mechanism for O-4 experiments.

This module provides the value-gated coupling mechanism that increases
SSC by modulating spatial arrangement based on semantic similarity.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def apply_value_gate(
    base_coords: np.ndarray,
    embeddings: np.ndarray,
    lam: float,
    seed: int = 42,
    radius: float = 1.0,
    layout: str = 'grid'
) -> np.ndarray:
    """
    Apply value gate parameter λ to arrange items spatially.
    
    Now supports morphing from a structured base_layout (e.g., grid)
    to a semantic layout via linear interpolation.

    λ=0: Returns base_coords exactly (Structure preserved).
    λ=1: Returns PCA-projected semantic coordinates (Meaning dominant).
    0<λ<1: Linear interpolation between Structure and Meaning.

    Parameters
    ----------
    base_coords : ndarray, shape (n_items, 2)
        Base coordinates (e.g., 8x8 grid). 
        MUST be provided and determines the starting structure at λ=0.
    embeddings : ndarray, shape (n_items, dim)
        Semantic embeddings.
    lam : float in [0, 1]
        Value gate parameter.
    seed : int
        Random seed (used for PCA projection).
    radius : float
        (Deprecated in this logic but kept for compatibility).
    layout : str
        (Deprecated in this logic but kept for compatibility).

    Returns
    -------
    coords_gated : ndarray, shape (n_items, 2)
        Spatially arranged coordinates under value gate.
    """
    if lam == 0.0:
        return base_coords.copy()

    pca = PCA(n_components=2, random_state=seed)
    coords_sem_raw = pca.fit_transform(embeddings)

    min_base = np.min(base_coords, axis=0)
    max_base = np.max(base_coords, axis=0)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    coords_sem_norm = scaler.fit_transform(coords_sem_raw)
    
    range_base = max_base - min_base
    coords_sem_scaled = coords_sem_norm * range_base + min_base

    coords_gated = (1 - lam) * base_coords + lam * coords_sem_scaled

    return coords_gated
