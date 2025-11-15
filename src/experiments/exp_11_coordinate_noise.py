"""EXP-11: Structural Stress Test (Coordinate Noise)

Tests O-3 (Stress Tolerance) by adding noise to spatial coordinates
while keeping semantic structure fixed. Examines whether coordinate
perturbations disrupt semantic-spatial relationships.

Key Finding:
    SSC ≈ 0 despite coordinate noise (confirms O-3 stress tolerance)

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
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.5]

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp11_coordinate_noise"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_coordinate_noise(coords: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add Gaussian noise to coordinates.
    
    Args:
        coords: Original coordinates
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        Noisy coordinates
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, coords.shape)
    return coords + noise


def run_single_trial(seed: int, noise_level: float) -> dict:
    """Run single trial with coordinate noise.
    
    Args:
        seed: Random seed for this trial
        noise_level: Noise level
        
    Returns:
        Dictionary with seed, noise_level, and SSC value
    """
    # Generate embeddings
    embeddings = generate_embeddings(N_ITEMS, DIM, seed)
    sem_dist = pdist(embeddings, 'correlation')
    
    # Circle arrangement with noise
    coords_original = generate_spatial_coords(N_ITEMS, 'circle', seed)
    coords_noisy = add_coordinate_noise(coords_original, noise_level, seed + 20000)
    spa_dist = pdist(coords_noisy, 'euclidean')
    
    # Compute SSC
    ssc = compute_ssc(sem_dist, spa_dist)
    
    return {
        'seed': seed,
        'noise_level': noise_level,
        'ssc': ssc
    }


def run_exp11():
    """Run complete EXP-11 experiment."""
    total_trials = len(NOISE_LEVELS) * N_TRIALS
    
    print("="*70)
    print("EXP-11: Structural Stress Test (Coordinate Noise, O-3)")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  NOISE_LEVELS = {NOISE_LEVELS}")
    print(f"  N_TRIALS = {N_TRIALS} per level")
    print(f"  TOTAL TRIALS = {total_trials}")
    print()
    
    # Set deterministic mode
    set_deterministic_mode()
    
    # Verify environment
    verify_environment(OUTPUT_DIR / "env.txt")
    
    # Run trials
    print("Running trials...")
    results = []
    for noise_level in NOISE_LEVELS:
        print(f"  Noise level = {noise_level:.2f}:")
        for i in range(N_TRIALS):
            seed = BASE_SEED + i
            trial_result = run_single_trial(seed, noise_level)
            results.append(trial_result)
            
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{N_TRIALS} trials")
        print()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute statistics by noise level
    stats_by_noise = {}
    for noise in NOISE_LEVELS:
        df_noise = df[df['noise_level'] == noise]
        ssc_values = df_noise['ssc'].values
        stats = compute_summary_stats(ssc_values)
        ci = bootstrap_ci(ssc_values, n_bootstrap=5000, seed=BASE_SEED)
        stats_by_noise[noise] = {
            **stats,
            'ci_90_lower': ci[0],
            'ci_90_upper': ci[1]
        }
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp11_coordinate_noise_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_11_coordinate_noise',
        'description': 'Stress tolerance test (O-3)',
        'parameters': {
            'n_items': N_ITEMS,
            'dim': DIM,
            'noise_levels': NOISE_LEVELS,
            'n_trials_per_level': N_TRIALS,
            'total_trials': total_trials,
            'base_seed': BASE_SEED
        },
        'results_by_noise': {str(k): v for k, v in stats_by_noise.items()}
    }
    
    json_path = OUTPUT_DIR / "exp11_coordinate_noise_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [stats_by_noise[n]['mean'] for n in NOISE_LEVELS]
    stds = [stats_by_noise[n]['std'] for n in NOISE_LEVELS]
    
    ax.errorbar(NOISE_LEVELS, means, yerr=stds, marker='o', linewidth=2,
                markersize=10, capsize=5, color='green', label='SSC')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5,
               label='SSC=0 (O-1 baseline)')
    ax.set_xlabel('Coordinate Noise Level', fontsize=12)
    ax.set_ylabel('SSC', fontsize=12)
    ax.set_title('EXP-11: SSC ≈ 0 Despite Coordinate Stress (O-3)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp11_coordinate_noise_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {OUTPUT_DIR / 'exp11_coordinate_noise_plot.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    for noise in NOISE_LEVELS:
        stats = stats_by_noise[noise]
        print(f"Noise = {noise:.2f}:")
        print(f"  SSC: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  90% CI: [{stats['ci_90_lower']:.4f}, {stats['ci_90_upper']:.4f}]")
    print()
    print("Interpretation:")
    print("  SSC ≈ 0 despite coordinate noise")
    print("  Confirms O-3: Robustness to spatial perturbations")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp11()


if __name__ == "__main__":
    main()
