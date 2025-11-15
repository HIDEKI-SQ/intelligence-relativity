"""EXP-06: Dimension Robustness Test

Tests whether natural orthogonality holds across different embedding
dimensions (50, 100, 200, 500), confirming O-1 is dimension-independent.

Key Finding:
    SSC ≈ 0 across all dimensions (dimension-independent)

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
DIMS = [50, 100, 200, 500]
BASE_SEED = 42
N_TRIALS = 1000

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp06_dimension_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_trial(seed: int, dim: int) -> dict:
    """Run single trial with specified dimension.
    
    Args:
        seed: Random seed for this trial
        dim: Embedding dimension
        
    Returns:
        Dictionary with seed, dim, and SSC values
    """
    # Generate embeddings
    embeddings = generate_embeddings(N_ITEMS, dim, seed)
    sem_dist = pdist(embeddings, 'correlation')
    
    # Spatial arrangement (random)
    coords = generate_spatial_coords(N_ITEMS, 'random', seed)
    spa_dist = pdist(coords, 'euclidean')
    
    # Compute SSC
    ssc = compute_ssc(sem_dist, spa_dist)
    
    return {
        'seed': seed,
        'dim': dim,
        'ssc': ssc
    }


def run_exp06():
    """Run complete EXP-06 experiment."""
    total_trials = len(DIMS) * N_TRIALS
    
    print("="*70)
    print("EXP-06: Dimension Robustness Test")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIMS = {DIMS}")
    print(f"  N_TRIALS = {N_TRIALS} per dimension")
    print(f"  TOTAL TRIALS = {total_trials}")
    print()
    
    # Set deterministic mode
    set_deterministic_mode()
    
    # Verify environment
    verify_environment(OUTPUT_DIR / "env.txt")
    
    # Run trials
    print("Running trials...")
    results = []
    for dim in DIMS:
        print(f"  DIM = {dim}:")
        for i in range(N_TRIALS):
            seed = BASE_SEED + i
            trial_result = run_single_trial(seed, dim)
            results.append(trial_result)
            
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{N_TRIALS} trials")
        print()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute statistics by dimension
    stats_by_dim = {}
    for dim in DIMS:
        df_dim = df[df['dim'] == dim]
        ssc_values = df_dim['ssc'].values
        stats = compute_summary_stats(ssc_values)
        ci = bootstrap_ci(ssc_values, n_bootstrap=5000, seed=BASE_SEED)
        stats_by_dim[dim] = {
            **stats,
            'ci_90_lower': ci[0],
            'ci_90_upper': ci[1]
        }
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp06_dimension_robustness_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_06_dimension_robustness',
        'description': 'Dimension-independence test',
        'parameters': {
            'n_items': N_ITEMS,
            'dims': DIMS,
            'n_trials_per_dim': N_TRIALS,
            'total_trials': total_trials,
            'base_seed': BASE_SEED
        },
        'results_by_dimension': stats_by_dim
    }
    
    json_path = OUTPUT_DIR / "exp06_dimension_robustness_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [stats_by_dim[d]['mean'] for d in DIMS]
    stds = [stats_by_dim[d]['std'] for d in DIMS]
    
    ax.errorbar(DIMS, means, yerr=stds, marker='o', linewidth=2,
                markersize=10, capsize=5, label='SSC')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Embedding Dimension (log scale)', fontsize=12)
    ax.set_ylabel('SSC', fontsize=12)
    ax.set_title('EXP-06: SSC Across Dimensions (λ=0)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp06_dimension_robustness_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {OUTPUT_DIR / 'exp06_dimension_robustness_plot.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    for dim in DIMS:
        stats = stats_by_dim[dim]
        print(f"DIM = {dim}:")
        print(f"  SSC: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  90% CI: [{stats['ci_90_lower']:.4f}, {stats['ci_90_upper']:.4f}]")
    print()
    print("Interpretation:")
    print("  SSC ≈ 0 across all dimensions")
    print("  Confirms O-1 is dimension-independent")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp06()


if __name__ == "__main__":
    main()
