"""Statistical utilities.

Includes bootstrap CI and simplified TOST-like equivalence test.

See README.md for implementation notes.
"""

import numpy as np
from typing import Dict, Tuple


def compute_summary_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics.
    
    Parameters
    ----------
    data : ndarray
        Data array
    
    Returns
    -------
    stats : dict
        Dictionary with mean, std, median, min, max, n
    
    Examples
    --------
    >>> stats = compute_summary_stats(ssc_values)
    >>> print(stats['mean'])
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data, ddof=1)),
        'median': float(np.median(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'n': int(len(data))
    }


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 5000,
    confidence: float = 0.90,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.
    
    Parameters
    ----------
    data : ndarray
        Data array
    n_bootstrap : int, default=5000
        Number of bootstrap samples
    confidence : float, default=0.90
        Confidence level (0.90 for 90% CI)
    seed : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    ci_lower, ci_upper : tuple
        Confidence interval bounds
    
    Examples
    --------
    >>> ci_lower, ci_upper = bootstrap_ci(ssc_values, n_bootstrap=5000)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    
    bootstrap_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(ci_lower), float(ci_upper)


def tost_equivalence(
    data: np.ndarray,
    null_value: float = 0.0,
    delta: float = 0.10
) -> Dict[str, float]:
    """
    Simplified TOST-like equivalence test.
    
    Tests whether mean is practically equivalent to null_value
    within margin delta.
    
    Parameters
    ----------
    data : ndarray
        Data array
    null_value : float, default=0.0
        Null hypothesis value
    delta : float, default=0.10
        Equivalence margin
    
    Returns
    -------
    result : dict
        TOST results including p-value and equivalence decision
    
    Notes
    -----
    Simplified implementation. For critical analyses, consider
    statsmodels.stats.weightstats.ttost_ind().
    
    Examples
    --------
    >>> result = tost_equivalence(ssc_values, null_value=0.0, delta=0.10)
    >>> print(result['equivalent'])
    True
    """
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    
    # Two one-sided tests
    t1 = (mean - (null_value - delta)) / se
    p1 = stats.t.cdf(t1, df=n-1)
    
    t2 = (mean - (null_value + delta)) / se
    p2 = 1 - stats.t.cdf(t2, df=n-1)
    
    # TOST p-value is max of two one-sided p-values
    p_tost = max(p1, p2)
    
    return {
        'mean': float(mean),
        'se': float(se),
        'delta': delta,
        'p_tost': float(p_tost),
        'equivalent': p_tost < 0.05,
        'ci_lower': float(null_value - delta),
        'ci_upper': float(null_value + delta)
    }
