"""EXP-04: 3D Cube Arrangement (Higher Spatial Dimensions)

Tests whether 3D spatial arrangement maintains natural orthogonality,
examining O-1 generality across spatial dimensionalities (1D, 2D, 3D).

Key Finding:
    SSC ≈ 0 in both 3D cube and random conditions (O-1 across dimensions)

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
import numpy as np

# === Configuration ===
N_ITEMS = 20
DIM = 100
BASE_SEED = 42
N_TRIALS = 1000

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp04_3d_cube_arrangement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_random_3d(n_items: int, seed: int) -> np.ndarray:
    """Generate random 3D coordinates.
    
    Args:
        n_items: Number of items
        seed: Random seed
        
    Returns:
        Random 3D coordinates (n_items, 3)
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, (n_items, 3))
    return coords


def run_single_trial(seed: int) -> dict:
    """Run single trial comparing 3D cube and random arrangements.
    
    Args:
        seed: Random seed for this trial
        
    Returns:
        Dictionary with seed and SSC values for both conditions
    """
    # Generate embeddings (shared)
    embeddings = generate_embeddings(N_ITEMS, DIM, seed)
    sem_dist = pdist(embeddings, 'correlation')
    
    # Condition 1: 3D Cube (grid)
    coords_cube = generate_spatial_coords(N_ITEMS, 'cube', seed)
    spa_dist_cube = pdist(coords_cube, 'euclidean')
    ssc_cube = compute_ssc(sem_dist, spa_dist_cube)
    
    # Condition 2: Random 3D
    coords_random = generate_random_3d(N_ITEMS, seed)
    spa_dist_random = pdist(coords_random, 'euclidean')
    ssc_random = compute_ssc(sem_dist, spa_dist_random)
    
    return {
        'seed': seed,
        'ssc_cube': ssc_cube,
        'ssc_random': ssc_random
    }


def run_exp04():
    """Run complete EXP-04 experiment."""
    print("="*70)
    print("EXP-04: 3D Cube Arrangement (Higher Spatial Dimensions)")
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
    stats_cube = compute_summary_stats(df['ssc_cube'].values)
    stats_random = compute_summary_stats(df['ssc_random'].values)
    
    ci_cube = bootstrap_ci(df['ssc_cube'].values, n_bootstrap=5000, seed=BASE_SEED)
    ci_random = bootstrap_ci(df['ssc_random'].values, n_bootstrap=5000, seed=BASE_SEED)
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp04_3d_cube_arrangement_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_04_3d_cube_arrangement',
        'description': 'Higher spatial dimensions test (3D cube)',
        'parameters': {
            'n_items': N_ITEMS,
            'dim': DIM,
            'n_trials': N_TRIALS,
            'base_seed': BASE_SEED
        },
        'results': {
            'cube3d': {
                **stats_cube,
                'ci_90_lower': ci_cube[0],
                'ci_90_upper': ci_cube[1]
            },
            'random3d': {
                **stats_random,
                'ci_90_lower': ci_random[0],
                'ci_90_upper': ci_random[1]
            }
        }
    }
    
    json_path = OUTPUT_DIR / "exp04_3d_cube_arrangement_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    means = {
        'Cube (3D)': stats_cube['mean'],
        'Random (3D)': stats_random['mean']
    }
    cis = {
        'Cube (3D)': ci_cube,
        'Random (3D)': ci_random
    }
    
    plot_ci(
        means,
        cis,
        title="EXP-04: 3D Cube vs Random (λ=0)",
        ylabel="SSC",
        output_path=OUTPUT_DIR / "exp04_3d_cube_arrangement_comparison.png"
    )
    print(f"  ✅ {OUTPUT_DIR / 'exp04_3d_cube_arrangement_comparison.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    print(f"SSC (cube):   {stats_cube['mean']:.4f} ± {stats_cube['std']:.4f}")
    print(f"  90% CI: [{ci_cube[0]:.4f}, {ci_cube[1]:.4f}]")
    print()
    print(f"SSC (random): {stats_random['mean']:.4f} ± {stats_random['std']:.4f}")
    print(f"  90% CI: [{ci_random[0]:.4f}, {ci_random[1]:.4f}]")
    print()
    print("Interpretation:")
    print("  Both conditions show SSC ≈ 0")
    print("  Confirms O-1 extends to higher spatial dimensions (3D)")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp04()


if __name__ == "__main__":
    main()
