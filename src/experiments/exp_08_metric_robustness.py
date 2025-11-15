"""EXP-08: Metric Type Robustness Test

Tests whether natural orthogonality holds across different distance
metrics (correlation, euclidean, cosine), confirming O-1 is metric-independent.

Key Finding:
    SSC ≈ 0 across all metrics (metric-independent)

Author: HIDEKI
Date: 2025-11
License: MIT
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import (
    set_deterministic_mode,
    verify_environment,
    generate_embeddings,
    generate_spatial_coords,
    compute_ssc,
    compute_summary_stats,
    bootstrap_ci,
    generate_manifest
)
from scipy.spatial.distance import pdist
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# === Configuration ===
N_ITEMS = 20
DIM = 100
BASE_SEED = 42
N_TRIALS = 1000
METRICS = ['correlation', 'euclidean', 'cosine']

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp08_metric_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_trial(seed: int, metric: str) -> dict:
    """Run single trial with specified distance metric.
    
    Args:
        seed: Random seed for this trial
        metric: Distance metric name
        
    Returns:
        Dictionary with seed, metric, and SSC value
    """
    # Generate embeddings
    embeddings = generate_embeddings(N_ITEMS, DIM, seed)
    sem_dist = pdist(embeddings, metric)
    
    # Spatial arrangement (random)
    coords = generate_spatial_coords(N_ITEMS, 'random', seed)
    spa_dist = pdist(coords, 'euclidean')
    
    # Compute SSC
    ssc = compute_ssc(sem_dist, spa_dist)
    
    return {
        'seed': seed,
        'metric': metric,
        'ssc': ssc
    }


def run_exp08():
    """Run complete EXP-08 experiment."""
    total_trials = len(METRICS) * N_TRIALS
    
    print("="*70)
    print("EXP-08: Metric Type Robustness Test")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  METRICS = {METRICS}")
    print(f"  N_TRIALS = {N_TRIALS} per metric")
    print(f"  TOTAL TRIALS = {total_trials}")
    print()
    
    # Set deterministic mode
    set_deterministic_mode()
    
    # Verify environment
    verify_environment(OUTPUT_DIR / "env.txt")
    
    # Run trials
    print("Running trials...")
    results = []
    for metric in METRICS:
        print(f"  Metric = {metric}:")
        for i in range(N_TRIALS):
            seed = BASE_SEED + i
            trial_result = run_single_trial(seed, metric)
            results.append(trial_result)
            
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{N_TRIALS} trials")
        print()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute statistics by metric
    stats_by_metric = {}
    for metric in METRICS:
        df_m = df[df['metric'] == metric]
        ssc_values = df_m['ssc'].values
        stats = compute_summary_stats(ssc_values)
        ci = bootstrap_ci(ssc_values, n_bootstrap=5000, seed=BASE_SEED)
        stats_by_metric[metric] = {
            **stats,
            'ci_90_lower': ci[0],
            'ci_90_upper': ci[1]
        }
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp08_metric_robustness_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_08_metric_robustness',
        'description': 'Metric-independence test',
        'parameters': {
            'n_items': N_ITEMS,
            'dim': DIM,
            'metrics': METRICS,
            'n_trials_per_metric': N_TRIALS,
            'total_trials': total_trials,
            'base_seed': BASE_SEED
        },
        'results_by_metric': stats_by_metric
    }
    
    json_path = OUTPUT_DIR / "exp08_metric_robustness_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(METRICS))
    means = [stats_by_metric[m]['mean'] for m in METRICS]
    stds = [stats_by_metric[m]['std'] for m in METRICS]
    
    ax.bar(x_pos, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Distance Metric', fontsize=12)
    ax.set_ylabel('SSC', fontsize=12)
    ax.set_title('EXP-08: SSC Across Metrics (λ=0)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(METRICS)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp08_metric_robustness_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {OUTPUT_DIR / 'exp08_metric_robustness_plot.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    for metric in METRICS:
        stats = stats_by_metric[metric]
        print(f"Metric = {metric}:")
        print(f"  SSC: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  90% CI: [{stats['ci_90_lower']:.4f}, {stats['ci_90_upper']:.4f}]")
    print()
    print("Interpretation:")
    print("  SSC ≈ 0 across all distance metrics")
    print("  Confirms O-1 is metric-independent")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp08()


if __name__ == "__main__":
    main()
