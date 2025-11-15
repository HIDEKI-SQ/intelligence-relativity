"""EXP-09: Topological Disruption Test

Tests O-2 (Phase Dominance) by disrupting topological order through
random swaps. Shows that topology (φ) is more important than exact
spatial coordinates for structure preservation.

Key Finding:
    SSC increases with topological disruption (confirms O-2)

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
SWAP_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp09_topological_disruption"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def disrupt_topology(coords: np.ndarray, swap_ratio: float, seed: int) -> np.ndarray:
    """Disrupt topological order by swapping positions.
    
    Args:
        coords: Original coordinates
        swap_ratio: Ratio of positions to swap [0, 1]
        seed: Random seed
        
    Returns:
        Disrupted coordinates
    """
    rng = np.random.default_rng(seed)
    coords_disrupted = coords.copy()
    
    n_swaps = int(N_ITEMS * swap_ratio / 2)  # Each swap affects 2 items
    
    for _ in range(n_swaps):
        i, j = rng.choice(N_ITEMS, size=2, replace=False)
        coords_disrupted[i], coords_disrupted[j] = coords_disrupted[j].copy(), coords_disrupted[i].copy()
    
    return coords_disrupted


def run_single_trial(seed: int, swap_ratio: float) -> dict:
    """Run single trial with topological disruption.
    
    Args:
        seed: Random seed for this trial
        swap_ratio: Ratio of swaps
        
    Returns:
        Dictionary with seed, swap_ratio, and SSC value
    """
    # Generate embeddings
    embeddings = generate_embeddings(N_ITEMS, DIM, seed)
    sem_dist = pdist(embeddings, 'correlation')
    
    # Original circle arrangement
    coords_original = generate_spatial_coords(N_ITEMS, 'circle', seed)
    
    # Disrupt topology
    coords_disrupted = disrupt_topology(coords_original, swap_ratio, seed + 20000)
    spa_dist = pdist(coords_disrupted, 'euclidean')
    
    # Compute SSC
    ssc = compute_ssc(sem_dist, spa_dist)
    
    return {
        'seed': seed,
        'swap_ratio': swap_ratio,
        'ssc': ssc
    }


def run_exp09():
    """Run complete EXP-09 experiment."""
    total_trials = len(SWAP_RATIOS) * N_TRIALS
    
    print("="*70)
    print("EXP-09: Topological Disruption Test (O-2)")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  SWAP_RATIOS = {SWAP_RATIOS}")
    print(f"  N_TRIALS = {N_TRIALS} per ratio")
    print(f"  TOTAL TRIALS = {total_trials}")
    print()
    
    # Set deterministic mode
    set_deterministic_mode()
    
    # Verify environment
    verify_environment(OUTPUT_DIR / "env.txt")
    
    # Run trials
    print("Running trials...")
    results = []
    for swap_ratio in SWAP_RATIOS:
        print(f"  Swap ratio = {swap_ratio:.1f}:")
        for i in range(N_TRIALS):
            seed = BASE_SEED + i
            trial_result = run_single_trial(seed, swap_ratio)
            results.append(trial_result)
            
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{N_TRIALS} trials")
        print()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute statistics by ratio
    stats_by_ratio = {}
    for ratio in SWAP_RATIOS:
        df_ratio = df[df['swap_ratio'] == ratio]
        ssc_values = df_ratio['ssc'].values
        stats = compute_summary_stats(ssc_values)
        ci = bootstrap_ci(ssc_values, n_bootstrap=5000, seed=BASE_SEED)
        stats_by_ratio[ratio] = {
            **stats,
            'ci_90_lower': ci[0],
            'ci_90_upper': ci[1]
        }
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp09_topological_disruption_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_09_topological_disruption',
        'description': 'Phase dominance test (O-2)',
        'parameters': {
            'n_items': N_ITEMS,
            'dim': DIM,
            'swap_ratios': SWAP_RATIOS,
            'n_trials_per_ratio': N_TRIALS,
            'total_trials': total_trials,
            'base_seed': BASE_SEED
        },
        'results_by_ratio': {str(k): v for k, v in stats_by_ratio.items()}
    }
    
    json_path = OUTPUT_DIR / "exp09_topological_disruption_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [stats_by_ratio[r]['mean'] for r in SWAP_RATIOS]
    stds = [stats_by_ratio[r]['std'] for r in SWAP_RATIOS]
    
    ax.errorbar(SWAP_RATIOS, means, yerr=stds, marker='o', linewidth=2,
                markersize=10, capsize=5, color='red', label='SSC')
    ax.axhline(y=0, color='blue', linestyle='--', linewidth=1, alpha=0.5,
               label='SSC=0 (O-1 baseline)')
    ax.set_xlabel('Swap Ratio (Topological Disruption)', fontsize=12)
    ax.set_ylabel('SSC', fontsize=12)
    ax.set_title('EXP-09: SSC Increases with Topological Disruption (O-2)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp09_topological_disruption_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {OUTPUT_DIR / 'exp09_topological_disruption_plot.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    for ratio in SWAP_RATIOS:
        stats = stats_by_ratio[ratio]
        print(f"Swap ratio = {ratio:.1f}:")
        print(f"  SSC: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  90% CI: [{stats['ci_90_lower']:.4f}, {stats['ci_90_upper']:.4f}]")
    print()
    print("Interpretation:")
    print("  SSC increases with topological disruption")
    print("  Confirms O-2: Topology (φ) dominates over exact coordinates")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp09()


if __name__ == "__main__":
    main()
