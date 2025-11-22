"""Compute SP and SSC metrics for dimensionality reduction evaluation.

Uses the core SP measurement system to evaluate how well each
DR method preserves the original structure.
"""

import numpy as np
import pandas as pd
from typing import Dict

from src.core_sp.sp_metrics import compute_sp_components
from src.core_sp.ssc_wrapper import compute_ssc


def compute_all_metrics(
    X_original: np.ndarray,
    embeddings: Dict[str, np.ndarray],
    layout_type: str = "cluster"
) -> pd.DataFrame:
    """
    Compute SP and SSC for all DR methods.
    
    Parameters
    ----------
    X_original : ndarray, shape (n_samples, n_features)
        Original high-dimensional data
    embeddings : dict
        Dictionary mapping method name to 2D coordinates
    layout_type : str, default="cluster"
        Layout type for SP computation
    
    Returns
    -------
    results : DataFrame
        Columns: method, sp_adj, sp_ord, sp_clu, sp_total, ssc
    
    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> embeddings = {'pca': np.random.randn(100, 2)}
    >>> results = compute_all_metrics(X, embeddings)
    >>> results.columns
    Index(['method', 'sp_adj', 'sp_ord', 'sp_clu', 'sp_total', 'ssc'], dtype='object')
    """
    print(f"\n📏 Computing SP and SSC metrics...")
    
    results = []
    
    for method, coords in embeddings.items():
        if coords is None:
            print(f"  ⏭️  {method}: skipped (not available)")
            continue
        
        print(f"  Computing {method}...")
        
        # Compute SP components
        sp_comps = compute_sp_components(
            base_coords=X_original,
            trans_coords=coords,
            layout_type=layout_type,
            k=4
        )
        
        # Compute SSC
        ssc = compute_ssc(
            embeddings=X_original,
            coords=coords,
            semantic_metric='euclidean',
            spatial_metric='euclidean'
        )
        
        results.append({
            'method': method,
            'sp_adj': sp_comps.sp_adj,
            'sp_ord': sp_comps.sp_ord,
            'sp_clu': sp_comps.sp_clu,
            'sp_total': sp_comps.total,
            'ssc': ssc
        })
        
        print(f"  ✅ {method}: SP={sp_comps.total:.3f}, SSC={ssc:.3f}")
    
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    # Test
    X = np.random.randn(100, 50)
    embeddings = {
        'test_method': np.random.randn(100, 2)
    }
    results = compute_all_metrics(X, embeddings)
    print(f"\nTest successful!")
    print(results)
