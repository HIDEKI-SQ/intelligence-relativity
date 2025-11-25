"""
SP51: Lambda Trade-off with Grid Layout (BERT Embeddings)

Demonstrates SSC↑ and SP↓ trade-off with real BERT embeddings
starting from structured (grid) layout.

Grid layout (8×7, N=52) with BERT pretrained embeddings.
Lambda sweep: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
Trials: 1000 per lambda

Author: HIDEKI
Date: 2025-11
License: MIT
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.ssc_wrapper import compute_ssc
from src.core_sp.value_gate import apply_value_gate
from src.experiments_sp.bert_utils import load_bert_embeddings


def generate_grid_layout_52() -> np.ndarray:
    """Generate 8x7 grid layout for 52 BERT items."""
    grid_x = 8
    grid_y = 7
    coords = []
    for i in range(grid_y):
        for j in range(grid_x):
            if len(coords) < 52:
                coords.append([j, i])
    return np.array(coords, dtype=np.float64)


def run_sp51():
    """Run SP51 experiment: BERT embeddings with grid layout."""
    
    # Parameters
    N_TRIALS = 1000
    LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    BASE_SEED = 5100
    
    # Output directory
    output_dir = Path("outputs_sp/sp51_lambda_tradeoff_grid_bert")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load BERT embeddings
    print("Loading BERT embeddings...")
    bert_data = load_bert_embeddings(seed=BASE_SEED)
    embeddings = bert_data["embeddings"]
    n_items = embeddings.shape[0]
    
    print(f"  Loaded {n_items} items with embeddings shape {embeddings.shape}")
    
    # Generate grid layout (instead of circle)
    base_coords = generate_grid_layout_52()
    print(f"  Grid layout: {base_coords.shape}")
    
    layout_type = "grid"  # Key difference from sp31
    
    # Storage
    all_results = []
    
    print(f"\nSP51: Grid Layout Lambda Trade-off (BERT)")
    print(f"N={n_items}, trials={N_TRIALS}")
    print(f"Lambda values: {LAMBDAS}")
    print("-" * 60)
    
    for lam in LAMBDAS:
        print(f"\nLambda = {lam:.1f}")
        
        lambda_results = []
        
        for trial in range(N_TRIALS):
            seed = BASE_SEED + trial
            
            # Apply value gate
            trans_coords = apply_value_gate(
                base_coords=base_coords.copy(),
                embeddings=embeddings,
                lam=lam,
                seed=seed,
                radius=1.0
            )
            
            # Compute metrics
            sp_total = compute_sp_total(
                base_coords=base_coords,
                trans_coords=trans_coords,
                layout_type=layout_type
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
    raw_path = output_dir / "sp51_raw.json"
    df_all.to_json(raw_path, orient='records', indent=2)
    print(f"\nRaw data saved: {raw_path}")
    
    # Save summary
    summary = df_all.groupby('lambda').agg({
        'sp_total': ['mean', 'std'],
        'ssc': ['mean', 'std']
    }).reset_index()
    summary.columns = ['lambda', 'sp_mean', 'sp_std', 'ssc_mean', 'ssc_std']
    
    summary_path = output_dir / "sp51_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")
    
    print("\nSP51 complete!")
    return summary


if __name__ == "__main__":
    run_sp51()
