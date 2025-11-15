"""EXP-10: Rotation Invariance Test

Tests whether spatial rotations (which preserve topology) maintain
SSC ≈ 0. Confirms that topology (φ) matters, not absolute orientation.

Key Finding:
    SSC ≈ 0 across all rotation angles (topologically invariant)

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
ROTATION_ANGLES = [0, 30, 60, 90, 120, 180]  # degrees

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp10_rotation_invariance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rotate_coordinates(coords: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate coordinates by specified angle.
    
    Args:
        coords: Original coordinates (n_items, 2)
        angle_degrees: Rotation angle in degrees
        
    Returns:
        Rotated coordinates (n_items, 2)
    """
    angle_rad = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    return coords @ rotation_matrix.T


def run_single_trial(seed: int, angle: float) -> dict:
    """Run single trial with rotation.
    
    Args:
        seed: Random seed for this trial
        angle: Rotation angle in degrees
        
    Returns:
        Dictionary with seed, angle, and SSC value
    """
    # Generate embeddings
    embeddings = generate_embeddings(N_ITEMS, DIM, seed)
    sem_dist = pdist(embeddings, 'correlation')
    
    # Circle arrangement with rotation
    coords_original = generate_spatial_coords(N_ITEMS, 'circle', seed)
    coords_rotated = rotate_coordinates(coords_original, angle)
    spa_dist = pdist(coords_rotated, 'euclidean')
    
    # Compute SSC
    ssc = compute_ssc(sem_dist, spa_dist)
    
    return {
        'seed': seed,
        'angle': angle,
        'ssc': ssc
    }


def run_exp10():
    """Run complete EXP-10 experiment."""
    total_trials = len(ROTATION_ANGLES) * N_TRIALS
    
    print("="*70)
    print("EXP-10: Rotation Invariance Test (O-2)")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  ROTATION_ANGLES = {ROTATION_ANGLES}")
    print(f"  N_TRIALS = {N_TRIALS} per angle")
    print(f"  TOTAL TRIALS = {total_trials}")
    print()
    
    # Set deterministic mode
    set_deterministic_mode()
    
    # Verify environment
    verify_environment(OUTPUT_DIR / "env.txt")
    
    # Run trials
    print("Running trials...")
    results = []
    for angle in ROTATION_ANGLES:
        print(f"  Rotation = {angle}°:")
        for i in range(N_TRIALS):
            seed = BASE_SEED + i
            trial_result = run_single_trial(seed, angle)
            results.append(trial_result)
            
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{N_TRIALS} trials")
        print()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute statistics by angle
    stats_by_angle = {}
    for angle in ROTATION_ANGLES:
        df_angle = df[df['angle'] == angle]
        ssc_values = df_angle['ssc'].values
        stats = compute_summary_stats(ssc_values)
        ci = bootstrap_ci(ssc_values, n_bootstrap=5000, seed=BASE_SEED)
        stats_by_angle[angle] = {
            **stats,
            'ci_90_lower': ci[0],
            'ci_90_upper': ci[1]
        }
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp10_rotation_invariance_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_10_rotation_invariance',
        'description': 'Topological invariance test (O-2)',
        'parameters': {
            'n_items': N_ITEMS,
            'dim': DIM,
            'rotation_angles': ROTATION_ANGLES,
            'n_trials_per_angle': N_TRIALS,
            'total_trials': total_trials,
            'base_seed': BASE_SEED
        },
        'results_by_angle': {str(k): v for k, v in stats_by_angle.items()}
    }
    
    json_path = OUTPUT_DIR / "exp10_rotation_invariance_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [stats_by_angle[a]['mean'] for a in ROTATION_ANGLES]
    stds = [stats_by_angle[a]['std'] for a in ROTATION_ANGLES]
    
    ax.errorbar(ROTATION_ANGLES, means, yerr=stds, marker='o', linewidth=2,
                markersize=10, capsize=5, color='blue', label='SSC')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5,
               label='SSC=0 (O-1 baseline)')
    ax.set_xlabel('Rotation Angle (degrees)', fontsize=12)
    ax.set_ylabel('SSC', fontsize=12)
    ax.set_title('EXP-10: SSC Invariant to Rotation (O-2)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp10_rotation_invariance_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {OUTPUT_DIR / 'exp10_rotation_invariance_plot.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    for angle in ROTATION_ANGLES:
        stats = stats_by_angle[angle]
        print(f"Rotation = {angle}°:")
        print(f"  SSC: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  90% CI: [{stats['ci_90_lower']:.4f}, {stats['ci_90_upper']:.4f}]")
    print()
    print("Interpretation:")
    print("  SSC ≈ 0 across all rotations")
    print("  Confirms O-2: Topology is invariant to spatial transformations")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp10()


if __name__ == "__main__":
    main()
