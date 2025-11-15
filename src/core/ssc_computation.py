"""SSC (Semantic-Spatial Correlation) computation.

This module provides the core measurement instrument validated in I-1.
All functions use condensed distance vectors for consistency.

Validated in I-1. See README.md for details.
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Tuple, Union


def compute_ssc(
    sem_condensed: np.ndarray,
    spa_condensed: np.ndarray,
    return_pvalue: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Compute Semantic-Spatial Correlation (SSC).
    
    Parameters
    ----------
    sem_condensed : ndarray, shape (n_pairs,)
        Condensed semantic distance vector from pdist
    spa_condensed : ndarray, shape (n_pairs,)
        Condensed spatial distance vector from pdist
    return_pvalue : bool, default=False
        If True, return (ssc, p_value) tuple
    
    Returns
    -------
    ssc : float or tuple
        Spearman correlation coefficient in [-1, 1]
        Returns (ssc, p_value) if return_pvalue=True
    
    Notes
    -----
    Natural Orthogonality (O-1): Under λ=0, SSC ≈ 0
    
    Examples
    --------
    >>> from scipy.spatial.distance import pdist
    >>> sem_dist = pdist(embeddings, 'correlation')
    >>> spa_dist = pdist(coords, 'euclidean')
    >>> ssc = compute_ssc(sem_dist, spa_dist)
    """
    # Validate input
    if sem_condensed.shape != spa_condensed.shape:
        raise ValueError(
            f"Distance vectors must have same shape. "
            f"Got sem: {sem_condensed.shape}, spa: {spa_condensed.shape}"
        )
    
    if sem_condensed.ndim != 1:
        raise ValueError(
            f"Expected 1D condensed vectors. Got {sem_condensed.ndim}D array. "
            f"Use pdist() to generate condensed vectors."
        )
    
    # Compute Spearman correlation
    rho, pval = spearmanr(sem_condensed, spa_condensed)
    
    # Handle NaN (constant vectors)
    if np.isnan(rho):
        rho, pval = 0.0, 1.0
    
    if return_pvalue:
        return float(rho), float(pval)
    else:
        return float(rho)


def compute_ssc_from_data(
    embeddings: np.ndarray,
    coords: np.ndarray,
    semantic_metric: str = 'correlation',
    spatial_metric: str = 'euclidean',
    return_pvalue: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Compute SSC directly from embeddings and coordinates.
    
    Convenience function that computes condensed distance vectors
    internally before calling compute_ssc().
    
    Parameters
    ----------
    embeddings : ndarray, shape (n_items, dim)
        Semantic embeddings
    coords : ndarray, shape (n_items, 2) or (n_items, 3)
        Spatial coordinates
    semantic_metric : str, default='correlation'
        Distance metric for embeddings
    spatial_metric : str, default='euclidean'
        Distance metric for coordinates
    return_pvalue : bool, default=False
        If True, return (ssc, p_value) tuple
    
    Returns
    -------
    ssc : float or tuple
        SSC value, or (ssc, p_value) if return_pvalue=True
    
    Examples
    --------
    >>> embeddings = generate_embeddings(64, 128, seed=42)
    >>> coords = generate_spatial_coords(64, 'circle', seed=42)
    >>> ssc = compute_ssc_from_data(embeddings, coords)
    """
    from scipy.spatial.distance import pdist
    
    # Generate condensed distance vectors
    sem_condensed = pdist(embeddings, metric=semantic_metric)
    spa_condensed = pdist(coords, metric=spatial_metric)
    
    # Compute SSC
    return compute_ssc(sem_condensed, spa_condensed, return_pvalue=return_pvalue)
