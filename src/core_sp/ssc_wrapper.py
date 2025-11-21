"""SSC computation wrapper for v2 compatibility.

This module provides a simplified interface to v1's compute_ssc_from_data,
maintaining compatibility while allowing v2-specific usage patterns.
"""

from src.core.ssc_computation import compute_ssc_from_data
import numpy as np
from typing import Union, Tuple


def compute_ssc(
    embeddings: np.ndarray,
    coords: np.ndarray,
    semantic_metric: str = 'correlation',
    spatial_metric: str = 'euclidean',
    return_pvalue: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Compute SSC (Semantic-Spatial Correlation) with simplified interface.
    
    This is a v2 wrapper around v1's compute_ssc_from_data for consistent
    usage across O-3 and O-4 experiments.
    
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
        SSC value in [-1, 1], or (ssc, p_value) if return_pvalue=True
    
    Notes
    -----
    Under λ=0 (natural orthogonality), SSC ≈ 0 (O-1).
    Under λ>0 (value-gated coupling), SSC increases with λ (O-4).
    
    Examples
    --------
    >>> from src.core_sp.generators import generate_semantic_embeddings
    >>> rng = np.random.default_rng(42)
    >>> embeddings = generate_semantic_embeddings(64, 128, rng)
    >>> coords = rng.uniform(-1, 1, (64, 2))
    >>> ssc = compute_ssc(embeddings, coords)
    >>> -1 <= ssc <= 1
    True
    """
    return compute_ssc_from_data(
        embeddings=embeddings,
        coords=coords,
        semantic_metric=semantic_metric,
        spatial_metric=spatial_metric,
        return_pvalue=return_pvalue
    )
