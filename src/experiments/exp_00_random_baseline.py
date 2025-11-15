"""EXP-00: Random Baseline (Natural Orthogonality)

Establishes baseline condition where items are arranged randomly
without spatial structure, serving as null hypothesis (λ=0) for
natural orthogonality observation (O-1).

Key Finding:
    SSC ≈ 0 in random condition (natural orthogonality)

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

# === Configuration ===
N_ITEMS = 20
DIM = 100
BASE_SEED = 42
N_TRIALS = 1000

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp00_random_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_trial(seed: int) -> dict:
    """Run single trial of random baseline experiment.
    
    Args:
        seed: Random seed for this trial
        
    Returns:
        Dictionary with seed and SSC value
    """
    # Generate data
    embeddings = generate_embeddings(N_ITEMS, DIM, seed)
    coords = generate_spatial_coords(N_ITEMS, 'random', seed)
    
    # Compute distances (condensed vectors)
    sem_dist = pdist(embeddings, 'correlation')
    spa_dist = pdist(coords, 'euclidean')
    
    # Compute SSC
    ssc = compute_ssc(sem_dist, spa_dist)
    
    return {
        'seed': seed,
        'ssc': ssc
    }


def run_exp00():
    """Run complete EXP-00 experiment."""
    print("="*70)
    print("EXP-00: Random Baseline (Natural Orthogonality)")
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
    ssc_values = df['ssc'].values
    
    # Compute statistics
    stats = compute_summary_stats(ssc_values)
    ci_lower, ci_upper = bootstrap_ci(ssc_values, n_bootstrap=5000, seed=BASE_SEED)
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp00_random_baseline_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_00_random_baseline',
        'description': 'Natural orthogonality baseline (λ=0)',
        'parameters': {
            'n_items': N_ITEMS,
            'dim': DIM,
            'n_trials': N_TRIALS,
            'base_seed': BASE_SEED
        },
        'results': {
            **stats,
            'ci_90_lower': ci_lower,
            'ci_90_upper': ci_upper
        }
    }
    
    json_path = OUTPUT_DIR / "exp00_random_baseline_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    plot_histogram(
        ssc_values,
        title="EXP-00: Random Baseline (λ=0)",
        xlabel="SSC",
        output_path=OUTPUT_DIR / "exp00_random_baseline_histogram.png"
    )
    print(f"  ✅ {OUTPUT_DIR / 'exp00_random_baseline_histogram.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    print(f"SSC (random): {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"90% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()
    print("Interpretation:")
    print("  SSC ≈ 0 confirms natural orthogonality (O-1)")
    print("  Structure and meaning are independent without value pressure (λ=0)")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp00()


if __name__ == "__main__":
    main()
