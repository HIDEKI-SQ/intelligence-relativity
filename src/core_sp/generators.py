"""Semantic embedding generation utilities.

This module provides common functions for generating semantic embeddings
and applying semantic noise, shared across O-3 and O-4 experiments.
"""

import numpy as np


def generate_semantic_embeddings(
    n_items: int,
    dim: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate random semantic embeddings with unit normalization.
    
    Parameters
    ----------
    n_items : int
        Number of items
    dim : int
        Embedding dimension
    rng : np.random.Generator
        Random number generator (for determinism)
    
    Returns
    -------
    embeddings : ndarray, shape (n_items, dim)
        Unit-normalized random embeddings from standard normal distribution
    
    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> embeddings = generate_semantic_embeddings(64, 128, rng)
    >>> embeddings.shape
    (64, 128)
    >>> np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    True
    """
    x = rng.normal(size=(n_items, dim))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


def add_semantic_noise(
    emb: np.ndarray,
    rng: np.random.Generator,
    sigma: float
) -> np.ndarray:
    """
    Add Gaussian noise to semantic embeddings and re-normalize.
    
    Parameters
    ----------
    emb : ndarray, shape (n_items, dim)
        Original embeddings
    rng : np.random.Generator
        Random number generator
    sigma : float
        Noise standard deviation
    
    Returns
    -------
    emb_noisy : ndarray, shape (n_items, dim)
        Noisy embeddings (re-normalized to unit length)
    
    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> emb = generate_semantic_embeddings(64, 128, rng)
    >>> emb_noisy = add_semantic_noise(emb, rng, sigma=0.1)
    >>> emb_noisy.shape
    (64, 128)
    """
    noise = rng.normal(loc=0.0, scale=sigma, size=emb.shape)
    x = emb + noise
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms
