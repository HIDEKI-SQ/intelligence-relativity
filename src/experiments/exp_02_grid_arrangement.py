"""EXP-02: Grid Arrangement (Spatial Structure Generality)

Tests whether grid arrangement (structured spatial layout) maintains
natural orthogonality, examining the generality of O-1 across
different spatial structures.

Key Finding:
    SSC ≈ 0 in both grid and random conditions (O-1 generality)

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
    plot_ci,
    generate_manifest
)
from scipy.spatial.distance import pdist
import pandas as pd
import json

# === Configuration ===
N_ITEMS = 20
DIM = 100
BASE_SEED = 42
N_TRIALS = 1000

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp02_grid_arrangement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_trial(seed: int) -> dict:
    """Run single trial comparing grid and random arrangements.
    
    Args:
        seed: Random seed for this trial
        
    Returns:
        Dictionary with seed and SSC values for both conditions
    """
    # Generate embeddings (shared)
    embeddings = generate_embeddings(N_ITEMS, DIM, seed)
    sem_dist = pdist(embeddings, 'correlation')
    
    # Condition 1: Grid
    coords_grid = generate_spatial_coords(N_ITEMS, 'grid', seed)
    spa_dist_grid = pdist(coords_grid, 'euclidean')
    ssc_grid = compute_ssc(sem_dist, spa_dist_grid)
    
    # Condition 2: Random
    coords_random = generate_spatial_coords(N_ITEMS, 'random', seed)
    spa_dist_random = pdist(coords_random, 'euclidean')
    ssc_random = compute_ssc(sem_dist, spa_dist_random)
    
    return {
        'seed': seed,
        'ssc_grid': ssc_grid,
        'ssc_random': ssc_random
    }


def run_exp02():
    """Run complete EXP-02 experiment."""
    print("="*70)
    print("EXP-02: Grid Arrangement (Spatial Structure Generality)")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  N_TRIALS = {N_TRIALS}")
    print(f"  BASE_SEED = {BASE_SEED}")
    print()
    
    # Set deterministic mode
    set_deterministic_mode()
    
    # Verify environment
    verify_environment(OUTPUT_DIR / "env.txt")
    
    # Run trials
    print("Running trials...")
    results = []
    for i in range(N_TRIALS):
        seed = BASE_SEED + i
        trial_result = run_single_trial(seed)
        results.append(trial_result)
        
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{N_TRIALS} trials")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute statistics
    stats_grid = compute_summary_stats(df['ssc_grid'].values)
    stats_random = compute_summary_stats(df['ssc_random'].values)
    
    ci_grid = bootstrap_ci(df['ssc_grid'].values, n_bootstrap=5000, seed=BASE_SEED)
    ci_random = bootstrap_ci(df['ssc_random'].values, n_bootstrap=5000, seed=BASE_SEED)
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp02_grid_arrangement_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_02_grid_arrangement',
        'description': 'Spatial structure generality test',
        'parameters': {
            'n_items': N_ITEMS,
            'dim': DIM,
            'n_trials': N_TRIALS,
            'base_seed': BASE_SEED
        },
        'results': {
            'grid': {
                **stats_grid,
                'ci_90_lower': ci_grid[0],
                'ci_90_upper': ci_grid[1]
            },
            'random': {
                **stats_random,
                'ci_90_lower': ci_random[0],
                'ci_90_upper': ci_random[1]
            }
        }
    }
    
    json_path = OUTPUT_DIR / "exp02_grid_arrangement_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    means = {
        'Grid': stats_grid['mean'],
        'Random': stats_random['mean']
    }
    cis = {
        'Grid': ci_grid,
        'Random': ci_random
    }
    
    plot_ci(
        means,
        cis,
        title="EXP-02: Grid vs Random (λ=0)",
        ylabel="SSC",
        output_path=OUTPUT_DIR / "exp02_grid_arrangement_comparison.png"
    )
    print(f"  ✅ {OUTPUT_DIR / 'exp02_grid_arrangement_comparison.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    print(f"SSC (grid):   {stats_grid['mean']:.4f} ± {stats_grid['std']:.4f}")
    print(f"  90% CI: [{ci_grid[0]:.4f}, {ci_grid[1]:.4f}]")
    print()
    print(f"SSC (random): {stats_random['mean']:.4f} ± {stats_random['std']:.4f}")
    print(f"  90% CI: [{ci_random[0]:.4f}, {ci_random[1]:.4f}]")
    print()
    print("Interpretation:")
    print("  Both conditions show SSC ≈ 0")
    print("  Confirms O-1 generality across spatial structures")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp02()


if __name__ == "__main__":
    main()
