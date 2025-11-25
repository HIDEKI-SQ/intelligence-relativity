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
    layout: str = 'grid'  # Kept for API compatibility; behavior depends on base_coords
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
    # 1. If lambda is 0, return base_coords immediately (Pure Structure preserved)
    # This ensures backward compatibility with experiments expecting the exact base layout.
    if lam == 0.0:
        return base_coords.copy()

    # 2. Generate "Semantic Coordinates" (Target for λ=1)
    # Project high-dimensional embeddings to 2D to define "where items want to go"
    # based purely on meaning.
    pca = PCA(n_components=2, random_state=seed)
    coords_sem_raw = pca.fit_transform(embeddings)

    # 3. Scale Semantic Coordinates to match Base Coordinates
    # This prevents the structure from "shrinking" or "exploding" during transition.
    # We align the min/max range of the semantic map to the base layout.
    
    # Get the range of the base coordinates
    min_base = np.min(base_coords, axis=0)
    max_base = np.max(base_coords, axis=0)
    
    # Scale semantic coordinates to 0-1 first
    scaler = MinMaxScaler(feature_range=(0, 1))
    coords_sem_norm = scaler.fit_transform(coords_sem_raw)
    
    # Then scale to the base coordinate range
    # Formula: (max - min) * value + min
    range_base = max_base - min_base
    coords_sem_scaled = coords_sem_norm * range_base + min_base

    # 4. Apply Linear Interpolation (Mixing)
    # coords = (1 - λ) * Structure + λ * Meaning
    coords_gated = (1 - lam) * base_coords + lam * coords_sem_scaled

    return coords_gated
