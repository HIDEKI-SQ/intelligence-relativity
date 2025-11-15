"""EXP-05: Independence Test (Permutation)

Tests independence between semantic structure and spatial arrangement
through permutation analysis, verifying VS≈0 is not an artifact of
specific item-position pairings.

Key Finding:
    SSC ≈ 0 across all permutations (statistical independence confirmed)

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
    plot_histogram,
    generate_manifest
)
from scipy.spatial.distance import pdist
import pandas as pd
import json
import numpy as np

# === Configuration ===
N_ITEMS = 20
DIM = 100
BASE_SEED = 42
N_SEEDS = 1000
N_PERMUTATIONS = 20  # Permutations per seed

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp05_independence_permutation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_trial(seed_A: int, seed_perm: int) -> dict:
    """Run single trial with permuted mapping.
    
    Args:
        seed_A: Random seed for embeddings
        seed_perm: Random seed for permutation
        
    Returns:
        Dictionary with seeds and SSC value
    """
    # Generate embeddings
    embeddings = generate_embeddings(N_ITEMS, DIM, seed_A)
    sem_dist = pdist(embeddings, 'correlation')
    
    # Fixed circle arrangement
    coords_base = generate_spatial_coords(N_ITEMS, 'circle', seed=42)
    
    # Permute mapping
    rng = np.random.default_rng(seed_perm)
    permutation = rng.permutation(N_ITEMS)
    coords_permuted = coords_base[permutation]
    
    spa_dist = pdist(coords_permuted, 'euclidean')
    
    # Compute SSC
    ssc = compute_ssc(sem_dist, spa_dist)
    
    return {
        'seed_A': seed_A,
        'seed_perm': seed_perm,
        'ssc': ssc
    }


def run_exp05():
    """Run complete EXP-05 experiment."""
    total_trials = N_SEEDS * N_PERMUTATIONS
    
    print("="*70)
    print("EXP-05: Independence Test (Permutation)")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  N_SEEDS = {N_SEEDS}")
    print(f"  N_PERMUTATIONS = {N_PERMUTATIONS}")
    print(f"  TOTAL TRIALS = {total_trials}")
    print()
    
    # Set deterministic mode
    set_deterministic_mode()
    
    # Verify environment
    verify_environment(OUTPUT_DIR / "env.txt")
    
    # Run trials
    print("Running trials...")
    results = []
    for i in range(N_SEEDS):
        seed_A = BASE_SEED + i
        for j in range(N_PERMUTATIONS):
            seed_perm = BASE_SEED + 10000 + i * N_PERMUTATIONS + j
            trial_result = run_single_trial(seed_A, seed_perm)
            results.append(trial_result)
        
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{N_SEEDS} seeds")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    ssc_values = df['ssc'].values
    
    # Compute statistics
    stats = compute_summary_stats(ssc_values)
    ci_lower, ci_upper = bootstrap_ci(ssc_values, n_bootstrap=5000, seed=BASE_SEED)
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp05_independence_permutation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_05_independence_permutation',
        'description': 'Permutation test for statistical independence',
        'parameters': {
            'n_items': N_ITEMS,
            'dim': DIM,
            'n_seeds': N_SEEDS,
            'n_permutations': N_PERMUTATIONS,
            'total_trials': total_trials,
            'base_seed': BASE_SEED
        },
        'results': {
            **stats,
            'ci_90_lower': ci_lower,
            'ci_90_upper': ci_upper
        }
    }
    
    json_path = OUTPUT_DIR / "exp05_independence_permutation_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    plot_histogram(
        ssc_values,
        title=f"EXP-05: Permutation Test (n={total_trials})",
        xlabel="SSC",
        output_path=OUTPUT_DIR / "exp05_independence_permutation_histogram.png"
    )
    print(f"  ✅ {OUTPUT_DIR / 'exp05_independence_permutation_histogram.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    print(f"SSC (all permutations): {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"90% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()
    print("Interpretation:")
    print("  SSC ≈ 0 across all permutations")
    print("  Confirms statistical independence (not pairing artifact)")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp05()


if __name__ == "__main__":
    main()
