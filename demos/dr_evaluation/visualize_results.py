"""Visualize dimensionality reduction evaluation results.

Creates comparison plots and 2D scatter plots for each method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional


def plot_comparison_bar(
    results: pd.DataFrame,
    output_path: Optional[Path] = None
) -> None:
    """
    Create bar chart comparing SP metrics across methods.
    
    Parameters
    ----------
    results : DataFrame
        Results from compute_all_metrics
    output_path : Path, optional
        If provided, save figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    methods = results['method'].values
    x = np.arange(len(methods))
    width = 0.2
    
    # SP components
    ax = axes[0]
    ax.bar(x - width*1.5, results['sp_adj'], width, label='SP_adj', alpha=0.8)
    ax.bar(x - width*0.5, results['sp_ord'], width, label='SP_ord', alpha=0.8)
    ax.bar(x + width*0.5, results['sp_clu'], width, label='SP_clu', alpha=0.8)
    ax.bar(x + width*1.5, results['sp_total'], width, label='SP_total', alpha=0.8)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('SP Score')
    ax.set_title('Structural Preservation (SP)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # SSC
    ax = axes[1]
    ax.bar(x, results['ssc'], width*2, alpha=0.8, color='coral')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('SSC')
    ax.set_title('Semantic-Spatial Correlation (SSC)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_embeddings_scatter(
    embeddings: Dict[str, np.ndarray],
    labels: np.ndarray,
    output_path: Optional[Path] = None
) -> None:
    """
    Create scatter plots of 2D embeddings.
    
    Parameters
    ----------
    embeddings : dict
        Dictionary mapping method name to 2D coordinates
    labels : ndarray
        Class labels for coloring
    output_path : Path, optional
        If provided, save figure to this path
    """
    n_methods = sum(1 for v in embeddings.values() if v is not None)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    ax_idx = 0
    for method, coords in embeddings.items():
        if coords is None:
            continue
        
        ax = axes[ax_idx]
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap='tab10',
            alpha=0.6, s=20
        )
        ax.set_title(f'{method.upper()}')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(alpha=0.3)
        
        ax_idx += 1
    
    # Add colorbar
    plt.colorbar(scatter, ax=axes, label='Class')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test
    results = pd.DataFrame({
        'method': ['tsne', 'umap', 'pca'],
        'sp_adj': [0.82, 0.85, 0.45],
        'sp_ord': [0.75, 0.78, 0.52],
        'sp_clu': [0.88, 0.92, 0.48],
        'sp_total': [0.82, 0.85, 0.48],
        'ssc': [0.08, 0.12, 0.35]
    })
    
    plot_comparison_bar(results)
    print("Test successful!")
