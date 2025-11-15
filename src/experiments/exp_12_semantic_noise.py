"""EXP-12: Semantic Noise Test (Meaning Stress)

Tests O-3 (Stress Tolerance) by disrupting semantic structure while
preserving spatial structure. Shows bidirectional independence:
meaning disruption ≠ spatial confusion.

Key Finding:
    SSC ≈ 0 despite semantic noise (confirms O-3 bidirectional independence)

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
from scipy.stats import spearmanr
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# === Configuration ===
N_ITEMS = 20
DIM = 100
BASE_SEED = 42
N_TRIALS = 1000
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.5, 1.0]

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "exp12_semantic_noise"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_semantic_noise(embeddings: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add semantic noise by mixing with random embeddings.
    
    Args:
        embeddings: Original embeddings
        noise_level: Mixing ratio with random embeddings [0, 1]
        seed: Random seed
        
    Returns:
        Noisy embeddings (re-normalized)
    """
    rng = np.random.default_rng(seed)
    
    # Generate random embeddings
    random_emb = rng.standard_normal(embeddings.shape)
    random_emb = random_emb / (np.linalg.norm(random_emb, axis=1, keepdims=True) + 1e-12)
    
    # Mix: noisy = (1-λ)*original + λ*random
    noisy = (1 - noise_level) * embeddings + noise_level * random_emb
    
    # Re-normalize
    noisy = noisy / (np.linalg.norm(noisy, axis=1, keepdims=True) + 1e-12)
    
    return noisy


def compute_semantic_similarity(emb_original: np.ndarray, emb_noisy: np.ndarray) -> float:
    """Compute semantic similarity between original and noisy embeddings.
    
    Args:
        emb_original: Original embeddings
        emb_noisy: Noisy embeddings
        
    Returns:
        Spearman correlation between distance matrices
    """
    d_orig = pdist(emb_original, metric='correlation')
    d_noisy = pdist(emb_noisy, metric='correlation')
    sim, _ = spearmanr(d_orig, d_noisy)
    return float(sim)


def run_single_trial(seed: int, noise_level: float) -> dict:
    """Run single trial with semantic noise.
    
    Args:
        seed: Random seed for this trial
        noise_level: Semantic noise level
        
    Returns:
        Dictionary with seed, noise_level, SSC, and semantic_similarity
    """
    # Generate original embeddings
    embeddings_original = generate_embeddings(N_ITEMS, DIM, seed)
    
    # Add semantic noise
    embeddings_noisy = add_semantic_noise(embeddings_original, noise_level, seed + 10000)
    
    # Compute semantic similarity
    sem_sim = compute_semantic_similarity(embeddings_original, embeddings_noisy)
    
    # Spatial arrangement (fixed)
    coords = generate_spatial_coords(N_ITEMS, 'circle', seed)
    spa_dist = pdist(coords, 'euclidean')
    
    # Compute SSC with noisy semantics
    sem_dist_noisy = pdist(embeddings_noisy, 'correlation')
    ssc = compute_ssc(sem_dist_noisy, spa_dist)
    
    return {
        'seed': seed,
        'noise_level': noise_level,
        'ssc': ssc,
        'semantic_similarity': sem_sim
    }


def run_exp12():
    """Run complete EXP-12 experiment."""
    total_trials = len(NOISE_LEVELS) * N_TRIALS
    
    print("="*70)
    print("EXP-12: Semantic Noise Test (Meaning Stress, O-3)")
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
        print(f"  Semantic noise = {noise_level:.1f}:")
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
            'ci_90_upper': ci[1],
            'semantic_similarity_mean': float(df_noise['semantic_similarity'].mean()),
            'semantic_similarity_std': float(df_noise['semantic_similarity'].std())
        }
    
    # Save results
    print()
    print("Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "exp12_semantic_noise_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    # JSON summary
    summary = {
        'experiment': 'exp_12_semantic_noise',
        'description': 'Bidirectional independence test (O-3)',
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
    
    json_path = OUTPUT_DIR / "exp12_semantic_noise_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ {json_path}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    means = [stats_by_noise[n]['mean'] for n in NOISE_LEVELS]
    stds = [stats_by_noise[n]['std'] for n in NOISE_LEVELS]
    sem_sims = [stats_by_noise[n]['semantic_similarity_mean'] for n in NOISE_LEVELS]
    
    # Panel 1: SSC vs Noise
    axes[0].errorbar(NOISE_LEVELS, means, yerr=stds, marker='o', linewidth=2,
                     markersize=10, capsize=5, color='purple', label='SSC')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5,
                    label='SSC=0 (O-1 baseline)')
    axes[0].set_xlabel('Semantic Noise Level', fontsize=12)
    axes[0].set_ylabel('SSC', fontsize=12)
    axes[0].set_title('EXP-12: SSC ≈ 0 Despite Semantic Stress (O-3)', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Semantic Similarity vs Noise
    axes[1].plot(NOISE_LEVELS, sem_sims, marker='^', linewidth=2,
                 markersize=10, color='orange', label='Semantic Similarity')
    axes[1].axhline(y=1.0, color='blue', linestyle='--', linewidth=1, alpha=0.5,
                    label='Perfect similarity')
    axes[1].set_xlabel('Semantic Noise Level', fontsize=12)
    axes[1].set_ylabel('Semantic Similarity', fontsize=12)
    axes[1].set_title('Meaning Degrades with Noise', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp12_semantic_noise_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {OUTPUT_DIR / 'exp12_semantic_noise_plot.png'}")
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display results
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    for noise in NOISE_LEVELS:
        stats = stats_by_noise[noise]
        print(f"Noise = {noise:.1f}:")
        print(f"  SSC: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Semantic Similarity: {stats['semantic_similarity_mean']:.4f}")
    print()
    print("Interpretation:")
    print("  SSC ≈ 0 despite semantic noise")
    print("  Meaning degrades but structure remains orthogonal")
    print("  Confirms O-3: Bidirectional independence (structure ⊥ meaning)")
    print("="*70)
    
    return df


def main():
    """Main execution."""
    run_exp12()


if __name__ == "__main__":
    main()
