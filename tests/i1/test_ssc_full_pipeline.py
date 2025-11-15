"""Test Suite 5: Full SSC Pipeline Cross-Implementation Validation

Validates the complete SSC computation pipeline including:
1. Distance computation (pdist) - Test Suites 1-4
2. Condensed vector extraction (squareform)
3. Rank correlation computation (spearmanr)

This extends I-1 validation to cover the entire measurement instrument.

Author: HIDEKI
Date: 2025-11
License: MIT
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import pandas as pd
import json


# === Configuration ===
N = 64          # Number of items
D = 128         # Embedding dimension
N_TRIALS = 1000 # Number of trials
BASE_SEED = 42  # Base random seed
METRICS = ['cosine', 'correlation', 'euclidean']
SPEC = 0.07     # Specification threshold

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "i1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# === Implementation 1: Standard (from core) ===
def compute_ssc_v1(sem_condensed: np.ndarray, spa_condensed: np.ndarray) -> float:
    """Standard implementation using condensed vectors."""
    rho, _ = spearmanr(sem_condensed, spa_condensed)
    return float(0.0 if np.isnan(rho) else rho)


# === Implementation 2: Direct from data ===
def compute_ssc_v2(
    embeddings: np.ndarray,
    coords: np.ndarray,
    semantic_metric: str,
    spatial_metric: str = 'euclidean'
) -> float:
    """Direct computation from raw data."""
    sem_condensed = pdist(embeddings, metric=semantic_metric)
    spa_condensed = pdist(coords, metric=spatial_metric)
    rho, _ = spearmanr(sem_condensed, spa_condensed)
    return float(0.0 if np.isnan(rho) else rho)


# === Implementation 3: Manual Spearman ===
def compute_ssc_v3(sem_condensed: np.ndarray, spa_condensed: np.ndarray) -> float:
    """Manual Spearman rank correlation (validates scipy.stats.spearmanr)."""
    n = len(sem_condensed)
    if n < 2:
        return 0.0
    
    # Manual rank computation
    rank_sem = np.argsort(np.argsort(sem_condensed)) + 1
    rank_spa = np.argsort(np.argsort(spa_condensed)) + 1
    
    # Spearman formula: 1 - 6*Σd²/(n(n²-1))
    d = rank_sem - rank_spa
    rho = 1.0 - (6.0 * np.sum(d**2)) / (n * (n**2 - 1))
    
    return float(rho)


# === Test Execution ===
def run_test_suite_5():
    """
    Run Test Suite 5: Full SSC Pipeline Cross-Implementation Test.
    
    Verifies that all 3 implementations produce identical SSC values
    within specification (|Δ| < 0.07, ideally < 10^-12).
    """
    print("="*70)
    print("Test Suite 5: Full SSC Pipeline Cross-Implementation Validation")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  N = {N} (items)")
    print(f"  D = {D} (dimension)")
    print(f"  N_TRIALS = {N_TRIALS}")
    print(f"  METRICS = {METRICS}")
    print(f"  SPEC = |Δ| < {SPEC}")
    print()
    
    # Storage
    results = {m: {'v1_v2': [], 'v1_v3': [], 'v2_v3': []} for m in METRICS}
    
    # Run trials
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(BASE_SEED + trial)
        
        # Generate test data
        embeddings = rng.standard_normal((N, D))
        coords = rng.uniform(-1, 1, (N, 2))
        
        for metric in METRICS:
            # Compute distances
            sem_condensed = pdist(embeddings, metric)
            spa_condensed = pdist(coords, 'euclidean')
            
            # Three implementations
            ssc_v1 = compute_ssc_v1(sem_condensed, spa_condensed)
            ssc_v2 = compute_ssc_v2(embeddings, coords, metric, 'euclidean')
            ssc_v3 = compute_ssc_v3(sem_condensed, spa_condensed)
            
            # Record differences
            results[metric]['v1_v2'].append(abs(ssc_v1 - ssc_v2))
            results[metric]['v1_v3'].append(abs(ssc_v1 - ssc_v3))
            results[metric]['v2_v3'].append(abs(ssc_v2 - ssc_v3))
        
        if (trial + 1) % 100 == 0:
            print(f"  Progress: {trial + 1}/{N_TRIALS} trials completed")
    
    print()
    
    # Analyze results
    print("="*70)
    print("Results: Cross-Implementation Differences")
    print("="*70)
    print()
    
    summary = []
    for metric in METRICS:
        print(f"Metric: {metric}")
        for comparison in ['v1_v2', 'v1_v3', 'v2_v3']:
            diffs = np.array(results[metric][comparison])
            mean_diff = diffs.mean()
            max_diff = diffs.max()
            p95_diff = np.percentile(diffs, 95)
            
            print(f"  {comparison}:")
            print(f"    Mean: {mean_diff:.2e}")
            print(f"    Max:  {max_diff:.2e}")
            print(f"    95th: {p95_diff:.2e}")
            
            summary.append({
                'metric': metric,
                'comparison': comparison,
                'mean_diff': float(mean_diff),
                'max_diff': float(max_diff),
                'p95_diff': float(p95_diff),
                'n_trials': N_TRIALS
            })
        print()
    
    # Overall assessment
    all_diffs = []
    for metric in METRICS:
        for comparison in ['v1_v2', 'v1_v3', 'v2_v3']:
            all_diffs.extend(results[metric][comparison])
    
    max_overall = np.max(all_diffs)
    
    print("="*70)
    print(f"Overall maximum difference: {max_overall:.2e}")
    print(f"Specification: |Δ| < {SPEC}")
    print(f"Status: {'✅ PASS' if max_overall < SPEC else '❌ FAIL'}")
    print("="*70)
    print()
    
    # Save results
    # CSV
    df = pd.DataFrame(summary)
    csv_path = OUTPUT_DIR / "test_suite_5_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")
    
    # JSON
    import platform
    import scipy
    
    json_data = {
        'test_suite': 5,
        'description': 'Full SSC Pipeline Cross-Implementation Validation',
        'parameters': {
            'N': N,
            'D': D,
            'n_trials': N_TRIALS,
            'metrics': METRICS,
            'base_seed': BASE_SEED
        },
        'results': {
            'max_overall_difference': float(max_overall),
            'specification': SPEC,
            'pass': bool(max_overall < SPEC),
            'by_metric': summary
        },
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'pandas': pd.__version__
        }
    }
    
    json_path = OUTPUT_DIR / "test_suite_5_summary.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✅ Saved: {json_path}")
    
    return max_overall < SPEC


def main():
    """Main execution."""
    success = run_test_suite_5()
    
    if success:
        print("\n🎉 Test Suite 5: PASSED")
        print("   All implementations agree within specification")
        return 0
    else:
        print("\n❌ Test Suite 5: FAILED")
        print("   Cross-implementation differences exceed specification")
        return 1


if __name__ == "__main__":
    sys.exit(main())
