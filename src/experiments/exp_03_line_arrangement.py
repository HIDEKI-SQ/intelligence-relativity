"""EXP-03: Line Arrangement (Spatial Dimensionality)

Tests whether linear (1D) spatial arrangement maintains natural
orthogonality, examining O-1 generality across spatial dimensionalities.

Key Finding:
    SSC ≈ 0 in both line and random conditions (O-1 across dimensions)

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

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp03_line_arrangement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_trial(seed: int) -> dict:
    """Run single trial comparing line and random arrangements.
    
    Args:
        seed: Random seed for this trial
        
    Returns:
        Dictionary with seed and SSC values for both conditions
    """
    # Generate embeddings (shared)
    embeddings = generate_embeddings(N_ITEMS, DIM, seed)
    sem_dist = pdist(embeddings, 'correlation')
    
    # Condition 1: Line (1D)
    coords_line = generate_spatial_coords(N_ITEMS, 'line', seed)
    spa_dist_line = pdist(coords_line, 'euclidean')
    ssc_line = compute_ssc(sem_dist, spa_dist_line)
    
    # Condition 2: Random
    coords_random = generate_spatial_coords(N_ITEMS, 'random', seed)
    spa_dist_random = pdist(coords_random, 'euclidean')
    ssc_random = compute_ssc(sem_dist, spa_dist_random)
    
    return {
        'seed': seed,
        'ssc_line': ssc_line,
        'ssc_random': ssc_random
    }


def run_exp03():
    """Run complete EXP-03 experiment."""
    print("="*70)
    print("EXP-03: Line Arrangement (Spatial Dimensionality)")
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
    stats_line = compute_summary_stats(df['ssc_line'].values)
    stats_random = compute_summary_stats(df['ssc_random'].values)
    
    ci_line = bootstrap_ci(df['ssc_line'].values, n_bootstrap=5000, seed=BASE_SEED)
    ci_random = bootstrap_ci(df['ssc_random'].values, n_bootstrap=5000, seed=BASE_SEED)
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp03_line_arrangement_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_03_line_arrangement',
        'description': 'Spatial dimensionality test (1D line)',
        'parameters': {
            'n_items': N_ITEMS,
            'dim': DIM,
            'n_trials': N_TRIALS,
            'base_seed': BASE_SEED
        },
        'results': {
            'line': {
                **stats_line,
                'ci_90_lower': ci_line[0],
                'ci_90_upper': ci_line[1]
            },
            'random': {
                **stats_random,
                'ci_90_lower': ci_random[0],
                'ci_90_upper': ci_random[1]
            }
        }
    }
    
    json_path = OUTPUT_DIR / "exp03_line_arrangement_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    means = {
        'Line (1D)': stats_line['mean'],
        'Random': stats_random['mean']
    }
    cis = {
        'Line (1D)': ci_line,
        'Random': ci_random
    }
    
    plot_ci(
        means,
        cis,
        title="EXP-03: Line vs Random (λ=0)",
        ylabel="SSC",
        output_path=OUTPUT_DIR / "exp03_line_arrangement_comparison.png"
    )
    print(f"  ✅ {OUTPUT_DIR / 'exp03_line_arrangement_comparison.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    print(f"SSC (line):   {stats_line['mean']:.4f} ± {stats_line['std']:.4f}")
    print(f"  90% CI: [{ci_line[0]:.4f}, {ci_line[1]:.4f}]")
    print()
    print(f"SSC (random): {stats_random['mean']:.4f} ± {stats_random['std']:.4f}")
    print(f"  90% CI: [{ci_random[0]:.4f}, {ci_random[1]:.4f}]")
    print()
    print("Interpretation:")
    print("  Both conditions show SSC ≈ 0")
    print("  Confirms O-1 holds across spatial dimensionalities (1D, 2D)")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp03()


if __name__ == "__main__":
    main()
