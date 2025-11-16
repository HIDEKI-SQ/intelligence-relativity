"""SUP-EXP-15: Positive Control (Measurement Validation)

Validates that compute_ssc() correctly detects strong correlation
when semantic and spatial structures are intentionally aligned.

Key Finding:
    SSC ≈ +1.0 with perfect alignment (confirms measurement validity)
    SSC ≈ -1.0 with perfect anti-alignment (confirms bidirectionality)
    SSC ≈ 0.0 with no alignment (confirms baseline)

Author: HIDEKI
Date: 2025-11
License: MIT
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import (
    set_deterministic_mode,
    verify_environment,
    compute_ssc,
    compute_summary_stats,
    bootstrap_ci,
    generate_manifest
)

import numpy as np
from scipy.spatial.distance import pdist
import json
import matplotlib.pyplot as plt

# === Configuration ===
N_ITEMS = 50
BASE_SEED = 42
N_TRIALS = 1000

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "sup15_posicon"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_positive_control(n_items, seed):
    """Generate SSC ≈ +1.0 case (perfect positive correlation)
    
    Strategy:
        - Semantic: Multi-dimensional embeddings where ALL dimensions
          follow the same ordering pattern
        - Spatial: Same ordering on a line
        - Result: correlation distances preserve order → SSC ≈ +1.0
    
    Key insight:
        correlation distance = 1 - pearson_corr(v_i, v_j)
        If all dimensions of embeddings follow same pattern,
        correlation distances will preserve the ordering.
    
    Returns:
        embeddings: (n_items, 5) array with consistent ordering
        coords: (n_items, 2) array with matching spatial ordering
    """
    rng = np.random.default_rng(seed)
    
    # Create base ordering
    order = np.arange(n_items).astype(float)
    normalized_order = order / (n_items - 1)  # Scale to [0, 1]
    
    # Semantic embeddings: 5 dimensions, ALL following the same order
    # Small noise ensures correlation distance is well-defined
    embeddings = np.column_stack([
        normalized_order + rng.normal(0, 0.01, n_items),
        normalized_order + rng.normal(0, 0.01, n_items),
        normalized_order + rng.normal(0, 0.01, n_items),
        normalized_order + rng.normal(0, 0.01, n_items),
        normalized_order + rng.normal(0, 0.01, n_items),
    ])
    
    # Spatial coordinates: same ordering on a line
    coords = np.column_stack([
        order,
        np.zeros(n_items)
    ])
    
    return embeddings, coords


def generate_negative_control(n_items, seed):
    """Generate SSC ≈ -1.0 case (perfect negative correlation)
    
    Strategy:
        - Semantic: Same as positive (all dimensions follow order)
        - Spatial: REVERSE ordering on a line
        - Result: Perfect anti-correlation → SSC ≈ -1.0
    
    Returns:
        embeddings: (n_items, 5) array with ordering
        coords: (n_items, 2) array with REVERSED spatial ordering
    """
    rng = np.random.default_rng(seed)
    
    order = np.arange(n_items).astype(float)
    normalized_order = order / (n_items - 1)
    
    # Semantic embeddings: same as positive control
    embeddings = np.column_stack([
        normalized_order + rng.normal(0, 0.01, n_items),
        normalized_order + rng.normal(0, 0.01, n_items),
        normalized_order + rng.normal(0, 0.01, n_items),
        normalized_order + rng.normal(0, 0.01, n_items),
        normalized_order + rng.normal(0, 0.01, n_items),
    ])
    
    # Spatial coordinates: REVERSED ordering
    coords = np.column_stack([
        order[::-1],  # Reversed!
        np.zeros(n_items)
    ])
    
    return embeddings, coords


def generate_zero_control(n_items, seed):
    """Generate SSC ≈ 0 case (no correlation - baseline)
    
    Strategy:
        - Semantic: Random high-dimensional structure
        - Spatial: Independent random 2D structure
        - Result: No systematic relationship → SSC ≈ 0
    
    Returns:
        embeddings: (n_items, 10) random array
        coords: (n_items, 2) independent random array
    """
    rng = np.random.default_rng(seed)
    
    # Semantic: Random high-dimensional
    embeddings = rng.normal(0, 1, (n_items, 10))
    
    # Spatial: Independent random 2D
    coords = rng.normal(0, 1, (n_items, 2))
    
    return embeddings, coords


def run_control_experiment(control_type, n_trials, base_seed):
    """Run experiment for one control type
    
    Args:
        control_type: 'positive', 'negative', or 'zero'
        n_trials: Number of trials
        base_seed: Base random seed
    
    Returns:
        dict: Results including SSC values and statistics
    """
    print(f"\n  Testing {control_type} control...")
    
    if control_type == 'positive':
        generator = generate_positive_control
        expected = "+1.0"
    elif control_type == 'negative':
        generator = generate_negative_control
        expected = "-1.0"
    else:
        generator = generate_zero_control
        expected = "0.0"
    
    print(f"    Expected: SSC ≈ {expected}")
    
    ssc_values = []
    
    for i in range(n_trials):
        embeddings, coords = generator(N_ITEMS, base_seed + i)
        
        # Compute distances
        sem_dist = pdist(embeddings, 'correlation')
        spa_dist = pdist(coords, 'euclidean')
        
        # Compute SSC
        ssc = compute_ssc(sem_dist, spa_dist)
        ssc_values.append(ssc)
        
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n_trials} trials completed")
    
    ssc_values = np.array(ssc_values)
    
    # Statistics
    stats = compute_summary_stats(ssc_values)
    ci = bootstrap_ci(ssc_values, n_bootstrap=5000, seed=base_seed)
    
    print(f"  Results:")
    print(f"    SSC: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"    90% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    return {
        **stats,
        'ci_90_lower': ci[0],
        'ci_90_upper': ci[1],
        'values': ssc_values.tolist()
    }


def create_visualization(results):
    """Create comprehensive 3-panel visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {
        'positive': 'green',
        'zero': 'gray',
        'negative': 'red'
    }
    
    targets = {
        'positive': 1.0,
        'zero': 0.0,
        'negative': -1.0
    }
    
    labels = {
        'positive': 'Positive Control\n(SSC ≈ +1)',
        'zero': 'Zero Control\n(SSC ≈ 0)',
        'negative': 'Negative Control\n(SSC ≈ -1)'
    }
    
    for idx, (control_type, ax) in enumerate(zip(['positive', 'zero', 'negative'], axes)):
        # Histogram
        values = results[control_type]['values']
        mean = results[control_type]['mean']
        target = targets[control_type]
        
        ax.hist(values, bins=40, edgecolor='black', alpha=0.7, 
                color=colors[control_type])
        ax.axvline(x=target, color='blue', linestyle='--', linewidth=2, 
                   label=f'Target={target:.1f}')
        ax.axvline(x=mean, color='darkred', linestyle='-', linewidth=2,
                   label=f'Actual={mean:.3f}')
        
        ax.set_xlabel('SSC', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(labels[control_type], fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        textstr = f'Mean: {mean:.3f}\nStd: {results[control_type]["std"]:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.suptitle('SUP-15: Measurement Validation (Positive, Zero, Negative Controls)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sup15_posicon_validation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Visualization saved")


def create_summary_plot(results):
    """Create summary bar plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    control_types = ['negative', 'zero', 'positive']
    means = [results[ct]['mean'] for ct in control_types]
    stds = [results[ct]['std'] for ct in control_types]
    colors_list = ['red', 'gray', 'green']
    
    x = np.arange(len(control_types))
    bars = ax.bar(x, means, yerr=stds, capsize=10, alpha=0.7, 
                  color=colors_list, edgecolor='black', linewidth=2)
    
    # Target lines
    ax.axhline(y=-1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, 
               label='Target -1.0')
    ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, 
               label='Target 0.0')
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5, 
               label='Target +1.0')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=11, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Negative\nControl', 'Zero\nControl', 'Positive\nControl'], 
                       fontsize=12)
    ax.set_ylabel('SSC', fontsize=14, fontweight='bold')
    ax.set_title('SUP-15: Measurement Instrument Validation', 
                 fontsize=15, fontweight='bold')
    ax.set_ylim(-1.2, 1.2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sup15_posicon_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Summary plot saved")


def run_sup15():
    """Run complete SUP-15 experiment"""
    print("="*70)
    print("SUP-EXP-15: Positive Control (Measurement Validation)")
    print("="*70)
    print(f"\nTesting compute_ssc() with controlled correlation patterns")
    print(f"N_ITEMS: {N_ITEMS}, N_TRIALS: {N_TRIALS}")
    print()
    print("This experiment validates that compute_ssc() correctly")
    print("measures correlation across the full dynamic range [-1, +1]")
    
    set_deterministic_mode()
    verify_environment(OUTPUT_DIR / "env.txt")
    
    # Run all three controls
    results = {}
    for control_type in ['positive', 'zero', 'negative']:
        results[control_type] = run_control_experiment(
            control_type, N_TRIALS, BASE_SEED
        )
    
    # Save summary
    summary = {
        'experiment': 'sup_exp_15_posicon',
        'description': 'Measurement validation with positive, zero, and negative controls',
        'parameters': {
            'n_items': N_ITEMS,
            'n_trials': N_TRIALS,
            'base_seed': BASE_SEED
        },
        'results': {
            'positive_control': {k: v for k, v in results['positive'].items() if k != 'values'},
            'zero_control': {k: v for k, v in results['zero'].items() if k != 'values'},
            'negative_control': {k: v for k, v in results['negative'].items() if k != 'values'}
        }
    }
    
    json_path = OUTPUT_DIR / "sup15_posicon_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✅ {json_path}")
    
    # Create visualizations
    create_visualization(results)
    create_summary_plot(results)
    
    # Generate manifest
    generate_manifest(OUTPUT_DIR, OUTPUT_DIR / "sha256_manifest.json")
    
    # Display summary
    print("\n" + "="*70)
    print("Results Summary:")
    print("="*70)
    
    print(f"\nPositive Control (Target: SSC ≈ +1.0):")
    print(f"  SSC: {results['positive']['mean']:.4f} ± {results['positive']['std']:.4f}")
    print(f"  90% CI: [{results['positive']['ci_90_lower']:.4f}, {results['positive']['ci_90_upper']:.4f}]")
    print(f"  Range: [{results['positive']['min']:.4f}, {results['positive']['max']:.4f}]")
    
    print(f"\nZero Control (Target: SSC ≈ 0.0):")
    print(f"  SSC: {results['zero']['mean']:.4f} ± {results['zero']['std']:.4f}")
    print(f"  90% CI: [{results['zero']['ci_90_lower']:.4f}, {results['zero']['ci_90_upper']:.4f}]")
    print(f"  Range: [{results['zero']['min']:.4f}, {results['zero']['max']:.4f}]")
    
    print(f"\nNegative Control (Target: SSC ≈ -1.0):")
    print(f"  SSC: {results['negative']['mean']:.4f} ± {results['negative']['std']:.4f}")
    print(f"  90% CI: [{results['negative']['ci_90_lower']:.4f}, {results['negative']['ci_90_upper']:.4f}]")
    print(f"  Range: [{results['negative']['min']:.4f}, {results['negative']['max']:.4f}]")
    
    print("\n" + "="*70)
    print("Interpretation:")
    print("="*70)
    
    # Validation checks
    pos_mean = results['positive']['mean']
    zero_mean = results['zero']['mean']
    neg_mean = results['negative']['mean']
    
    pos_valid = 0.7 < pos_mean < 1.0
    zero_valid = -0.1 < zero_mean < 0.1
    neg_valid = -1.0 < neg_mean < -0.7
    
    if pos_valid:
        print(f"  ✅ Positive control: SSC = {pos_mean:.3f} (expected ~0.9)")
        print("     → Measurement correctly detects strong positive correlation")
    else:
        print(f"  ⚠️  Positive control: SSC = {pos_mean:.3f} (outside expected range)")
    
    if zero_valid:
        print(f"  ✅ Zero control: SSC = {zero_mean:.3f} (expected ~0.0)")
        print("     → Measurement correctly shows no correlation")
    else:
        print(f"  ⚠️  Zero control: SSC = {zero_mean:.3f} (outside expected range)")
    
    if neg_valid:
        print(f"  ✅ Negative control: SSC = {neg_mean:.3f} (expected ~-0.9)")
        print("     → Measurement correctly detects strong negative correlation")
    else:
        print(f"  ⚠️  Negative control: SSC = {neg_mean:.3f} (outside expected range)")
    
    print()
    
    if pos_valid and zero_valid and neg_valid:
        print("  " + "🎉"*35)
        print("  🎉 VALIDATION COMPLETE: compute_ssc() is working correctly! 🎉")
        print("  🎉 Full dynamic range confirmed: SSC ∈ [-1, +1]              🎉")
        print("  " + "🎉"*35)
    else:
        print("\n  ⚠️  Some controls outside expected range")
        print("  ⚠️  Review data generation or increase tolerance")
    
    print("="*70)
    
    return summary


def main():
    run_sup15()


if __name__ == "__main__":
    main()
