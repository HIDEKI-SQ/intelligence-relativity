"""Apply dimensionality reduction methods (t-SNE, UMAP, PCA).

Provides unified interface for comparing different DR algorithms.
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict


def apply_all_methods(
    X: np.ndarray,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Apply t-SNE, UMAP, and PCA to input data.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        High-dimensional input data
    random_state : int, default=42
        Random seed for reproducibility
    
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
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    embeddings['tsne'] = tsne.fit_transform(X)
    print(f"  ✅ t-SNE completed")
    
    # UMAP
    print(f"  Running UMAP...")
    try:
        from umap import UMAP
        umap = UMAP(n_components=2, random_state=random_state, n_neighbors=15)
        embeddings['umap'] = umap.fit_transform(X)
        print(f"  ✅ UMAP completed")
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
