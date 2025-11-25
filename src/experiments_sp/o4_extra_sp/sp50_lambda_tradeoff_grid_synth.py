"""
SP50: Lambda Trade-off with Grid Layout (Synthetic)

Demonstrates SSC↑ and SP↓ trade-off when starting from structured layout.
Grid layout (8×8, N=64) with synthetic embeddings.
Lambda sweep: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
Trials: 1000 per lambda
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.core_sp.generators import generate_semantic_embeddings
from src.core_sp.value_gate import apply_value_gate
from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.ssc_wrapper import compute_ssc
from src.core_sp.deterministic import set_seed


def generate_grid_layout(n_items: int = 64) -> np.ndarray:
    """Generate 8x8 grid layout."""
    grid_size = int(np.sqrt(n_items))
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            coords.append([i, j])
    return np.array(coords, dtype=np.float64)


def run_sp50():
    """Run SP50 experiment."""
    
    # Parameters
    N = 64
    D = 128
    LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    N_TRIALS = 1000
    BASE_SEED = 5000
    
    # Output directory
    output_dir = Path("outputs_sp/sp50_lambda_tradeoff_grid_synth")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate base grid layout
    base_coords = generate_grid_layout(N)
    
    # Storage
    all_results = []
    
    print(f"SP50: Grid Layout Lambda Trade-off (Synthetic)")
    print(f"N={N}, D={D}, trials={N_TRIALS}")
    print(f"Lambda values: {LAMBDAS}")
    print("-" * 60)
    
    for lam in LAMBDAS:
        print(f"\nLambda = {lam:.1f}")
        
        lambda_results = []
        
        for trial in range(N_TRIALS):
            seed = BASE_SEED + trial
            set_seed(seed)
            
            # Generate semantic embeddings
            embeddings = generate_semantic_embeddings(N, D, seed=seed)
            
            # Apply value gate
            trans_coords = apply_value_gate(
                base_coords=base_coords.copy(),
                embeddings=embeddings,
                lam=lam,
                seed=seed
            )
            
            # Compute metrics
            sp_total = compute_sp_total(
                base_coords=base_coords,
                trans_coords=trans_coords,
                layout_type="grid"
            )
            
            ssc = compute_ssc(
                embeddings=embeddings,
                coords=trans_coords
            )
            
            # Store
            result = {
                'lambda': lam,
                'trial': trial,
                'seed': seed,
                'sp_total': sp_total,
                'ssc': ssc
            }
            lambda_results.append(result)
            all_results.append(result)
        
        # Summary for this lambda
        df_lam = pd.DataFrame(lambda_results)
        mean_sp = df_lam['sp_total'].mean()
        std_sp = df_lam['sp_total'].std()
        mean_ssc = df_lam['ssc'].mean()
        std_ssc = df_lam['ssc'].std()
        
        print(f"  SP:  {mean_sp:.4f} ± {std_sp:.4f}")
        print(f"  SSC: {mean_ssc:.4f} ± {std_ssc:.4f}")
    
    # Save raw data
    df_all = pd.DataFrame(all_results)
    raw_path = output_dir / "sp50_raw.json"
    df_all.to_json(raw_path, orient='records', indent=2)
    print(f"\nRaw data saved: {raw_path}")
    
    # Save summary
    summary = df_all.groupby('lambda').agg({
        'sp_total': ['mean', 'std'],
        'ssc': ['mean', 'std']
    }).reset_index()
    summary.columns = ['lambda', 'sp_mean', 'sp_std', 'ssc_mean', 'ssc_std']
    
    summary_path = output_dir / "sp50_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")
    
    print("\nSP50 complete!")
    return summary


if __name__ == "__main__":
    run_sp50()
