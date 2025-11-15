"""EXP-Beta: Initial Exploration (Historical)

Initial exploratory experiment that preceded EXP-00. Preserved for
transparency and completeness. For rigorous analysis, refer to EXP-00.

Key Finding:
    Initial observation of SSC ≈ 0 pattern (led to systematic investigation)

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
N_ITEMS = 20
DIM = 100
BASE_SEED = 42
N_TRIALS = 1000

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp_beta_initial_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_trial(seed: int) -> dict:
    """Run single trial.
    
    Args:
        seed: Random seed for this trial
        
    Returns:
        Dictionary with seed and SSC value
    """
    # Generate embeddings
    embeddings = generate_embeddings(N_ITEMS, DIM, seed)
    sem_dist = pdist(embeddings, 'correlation')
    
    # Random arrangement
    coords = generate_spatial_coords(N_ITEMS, 'random', seed)
    spa_dist = pdist(coords, 'euclidean')
    
    # Compute SSC
    ssc = compute_ssc(sem_dist, spa_dist)
    
    return {
        'seed': seed,
        'ssc': ssc
    }


def run_exp_beta():
    """Run complete EXP-Beta experiment."""
    print("="*70)
    print("EXP-Beta: Initial Exploration (Historical)")
    print("="*70)
    print()
    print("Note: Initial exploratory experiment. See EXP-00 for refined method.")
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
    stats = compute_summary_stats(df['ssc'].values)
    ci = bootstrap_ci(df['ssc'].values, n_bootstrap=5000, seed=BASE_SEED)
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp_beta_initial_exploration_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_beta_initial_exploration',
        'description': 'Initial exploratory experiment (historical)',
        'parameters': {
            'n_items': N_ITEMS,
            'dim': DIM,
            'n_trials': N_TRIALS,
            'base_seed': BASE_SEED
        },
        'results': {
            **stats,
            'ci_90_lower': ci[0],
            'ci_90_upper': ci[1]
        },
        'note': 'See EXP-00 for refined methodology'
    }
    
    json_path = OUTPUT_DIR / "exp_beta_initial_exploration_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Simple visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['ssc'], bins=15, edgecolor='black', alpha=0.7, color='lightblue')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='SSC=0')
    ax.axvline(x=stats['mean'], color='blue', linestyle='-', linewidth=2,
               label=f'Mean={stats["mean"]:.3f}')
    ax.set_xlabel('SSC')
    ax.set_ylabel('Frequency')
    ax.set_title('EXP-Beta: Initial Exploration (Historical)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp_beta_initial_exploration_histogram.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {OUTPUT_DIR / 'exp_beta_initial_exploration_histogram.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    print(f"SSC: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"90% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print()
    print("Historical Note:")
    print("  Initial observation of SSC ≈ 0 pattern")
    print("  Led to systematic investigation in EXP-00 through EXP-13")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp_beta()


if __name__ == "__main__":
    main()
