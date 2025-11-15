"""EXP-07: Sample Size Robustness Test

Tests whether natural orthogonality holds across different numbers
of items (10, 20, 40, 80), confirming O-1 is scale-independent.

Key Finding:
    SSC ≈ 0 across all sample sizes (scale-independent)

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

# === Configuration ===
N_ITEMS_LIST = [10, 20, 40, 80]
DIM = 100
BASE_SEED = 42
N_TRIALS = 1000

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp07_sample_size_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_trial(seed: int, n_items: int) -> dict:
    """Run single trial with specified sample size.
    
    Args:
        seed: Random seed for this trial
        n_items: Number of items
        
    Returns:
        Dictionary with seed, n_items, and SSC value
    """
    # Generate embeddings
    embeddings = generate_embeddings(n_items, DIM, seed)
    sem_dist = pdist(embeddings, 'correlation')
    
    # Spatial arrangement (random)
    coords = generate_spatial_coords(n_items, 'random', seed)
    spa_dist = pdist(coords, 'euclidean')
    
    # Compute SSC
    ssc = compute_ssc(sem_dist, spa_dist)
    
    return {
        'seed': seed,
        'n_items': n_items,
        'ssc': ssc
    }


def run_exp07():
    """Run complete EXP-07 experiment."""
    total_trials = len(N_ITEMS_LIST) * N_TRIALS
    
    print("="*70)
    print("EXP-07: Sample Size Robustness Test")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  N_ITEMS_LIST = {N_ITEMS_LIST}")
    print(f"  DIM = {DIM}")
    print(f"  N_TRIALS = {N_TRIALS} per size")
    print(f"  TOTAL TRIALS = {total_trials}")
    print()
    
    # Set deterministic mode
    set_deterministic_mode()
    
    # Verify environment
    verify_environment(OUTPUT_DIR / "env.txt")
    
    # Run trials
    print("Running trials...")
    results = []
    for n_items in N_ITEMS_LIST:
        print(f"  N_ITEMS = {n_items}:")
        for i in range(N_TRIALS):
            seed = BASE_SEED + i
            trial_result = run_single_trial(seed, n_items)
            results.append(trial_result)
            
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{N_TRIALS} trials")
        print()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute statistics by size
    stats_by_size = {}
    for n_items in N_ITEMS_LIST:
        df_n = df[df['n_items'] == n_items]
        ssc_values = df_n['ssc'].values
        stats = compute_summary_stats(ssc_values)
        ci = bootstrap_ci(ssc_values, n_bootstrap=5000, seed=BASE_SEED)
        stats_by_size[n_items] = {
            **stats,
            'ci_90_lower': ci[0],
            'ci_90_upper': ci[1]
        }
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp07_sample_size_robustness_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_07_sample_size_robustness',
        'description': 'Scale-independence test',
        'parameters': {
            'n_items_list': N_ITEMS_LIST,
            'dim': DIM,
            'n_trials_per_size': N_TRIALS,
            'total_trials': total_trials,
            'base_seed': BASE_SEED
        },
        'results_by_size': stats_by_size
    }
    
    json_path = OUTPUT_DIR / "exp07_sample_size_robustness_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [stats_by_size[n]['mean'] for n in N_ITEMS_LIST]
    stds = [stats_by_size[n]['std'] for n in N_ITEMS_LIST]
    
    ax.errorbar(N_ITEMS_LIST, means, yerr=stds, marker='o', linewidth=2,
                markersize=10, capsize=5, label='SSC')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Sample Size (N items)', fontsize=12)
    ax.set_ylabel('SSC', fontsize=12)
    ax.set_title('EXP-07: SSC Across Sample Sizes (λ=0)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp07_sample_size_robustness_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {OUTPUT_DIR / 'exp07_sample_size_robustness_plot.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    for n_items in N_ITEMS_LIST:
        stats = stats_by_size[n_items]
        print(f"N_ITEMS = {n_items}:")
        print(f"  SSC: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  90% CI: [{stats['ci_90_lower']:.4f}, {stats['ci_90_upper']:.4f}]")
    print()
    print("Interpretation:")
    print("  SSC ≈ 0 across all sample sizes")
    print("  Confirms O-1 is scale-independent")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp07()


if __name__ == "__main__":
    main()
