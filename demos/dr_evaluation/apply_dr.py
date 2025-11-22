"""Apply dimensionality reduction methods (t-SNE, UMAP, PCA).

Provides unified interface for comparing different DR algorithms.
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict


def apply_all_methods(
    X: np.ndarray,
    random_state: int = 42,
    perplexity: int = None
) -> Dict[str, np.ndarray]:
    """
    Apply t-SNE, UMAP, and PCA to input data.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        High-dimensional input data
    random_state : int, default=42
        Random seed for reproducibility
    perplexity : int, optional
        t-SNE perplexity parameter. If None, auto-adjusted based on sample size.
    
    Returns
    -------
    embeddings : dict
        Dictionary mapping method name to 2D coordinates
        Keys: 'tsne', 'umap', 'pca'
        Values: ndarray, shape (n_samples, 2)
    
    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> embeddings = apply_all_methods(X)
    >>> embeddings['tsne'].shape
    (100, 2)
    """
    print(f"\n🔄 Applying dimensionality reduction...")
    embeddings = {}
    
    # t-SNE
    print(f"  Running t-SNE...")
    # Auto-adjust perplexity based on sample size
    n_samples = X.shape[0]
    if perplexity is None:
        perplexity = min(30, max(5, n_samples // 3))
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    embeddings['tsne'] = tsne.fit_transform(X)
    print(f"  ✅ t-SNE completed (perplexity={perplexity})")
    
    # UMAP
    print(f"  Running UMAP...")
    try:
        from umap import UMAP
        # Auto-adjust n_neighbors based on sample size
        n_neighbors = min(15, max(2, n_samples // 5))
        umap = UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors)
        embeddings['umap'] = umap.fit_transform(X)
        print(f"  ✅ UMAP completed (n_neighbors={n_neighbors})")
    except ImportError:
        print(f"  ⚠️  UMAP not available (install umap-learn)")
        embeddings['umap'] = None
    
    # PCA
    print(f"  Running PCA...")
    pca = PCA(n_components=2, random_state=random_state)
    embeddings['pca'] = pca.fit_transform(X)
    print(f"  ✅ PCA completed")
    
    return embeddings


if __name__ == "__main__":
    # Test
    X = np.random.randn(100, 50)
    embeddings = apply_all_methods(X)
    
    print(f"\nTest successful!")
    for method, coords in embeddings.items():
        if coords is not None:
            print(f"{method}: {coords.shape}")
