"""Visualization utilities.

Standard plotting functions for E8 experiments and observations.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple


def plot_histogram(
    data: np.ndarray,
    title: str,
    xlabel: str = "SSC",
    output_path: Optional[Path] = None,
    bins: int = 30
) -> None:
    """
    Plot histogram with mean line.
    
    Parameters
    ----------
    data : ndarray
        Data to plot
    title : str
        Plot title
    xlabel : str, default="SSC"
        X-axis label
    output_path : Path, optional
        If provided, save figure to this path
    bins : int, default=30
        Number of histogram bins
    
    Examples
    --------
    >>> plot_histogram(ssc_values, "exp_00", output_path=Path("hist.png"))
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(data), color='red', linestyle='--', 
               linewidth=2, label=f'Mean={np.mean(data):.4f}')
    ax.axvline(0, color='blue', linestyle='--', 
               linewidth=1, alpha=0.5, label='Zero')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str = "Semantic Distance",
    ylabel: str = "Spatial Distance",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot scatter with correlation.
    
    Parameters
    ----------
    x, y : ndarray
        Data arrays
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    output_path : Path, optional
        If provided, save figure to this path
    
    Examples
    --------
    >>> plot_scatter(sem_dist, spa_dist, "Pairwise", output_path=Path("scatter.png"))
    """
    from scipy.stats import spearmanr
    
    rho, pval = spearmanr(x, y)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.5, s=30)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nρ={rho:.4f}, p={pval:.4f}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ci(
    means: Dict[str, float],
    cis: Dict[str, Tuple[float, float]],
    title: str,
    ylabel: str = "SSC",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot means with confidence intervals.
    
    Parameters
    ----------
    means : dict
        Mapping from label to mean value
    cis : dict
        Mapping from label to (ci_lower, ci_upper) tuple
    title : str
        Plot title
    ylabel : str, default="SSC"
        Y-axis label
    output_path : Path, optional
        If provided, save figure to this path
    
    Examples
    --------
    >>> means = {'cosine': -0.02, 'correlation': 0.01}
    >>> cis = {'cosine': (-0.05, 0.01), 'correlation': (-0.02, 0.04)}
    >>> plot_ci(means, cis, "Multi-metric SSC", output_path=Path("ci.png"))
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = list(means.keys())
    x_pos = np.arange(len(labels))
    
    for i, label in enumerate(labels):
        mean = means[label]
        ci_lower, ci_upper = cis[label]
        err_lower = mean - ci_lower
        err_upper = ci_upper - mean
        
        ax.errorbar(x_pos[i], mean, 
                   yerr=[[err_lower], [err_upper]],
                   marker='o', markersize=8, capsize=5,
                   label=label)
    
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
