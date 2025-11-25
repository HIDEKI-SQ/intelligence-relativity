"""SP-50: λ Sweep with Grid Layout (Synthetic Embeddings).
O-4 Value-Gated Coupling: Grid layout variant demonstrating SSC-SP tradeoff
with synthetic embeddings on structured (8×8 grid) initial configuration.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.ssc_wrapper import compute_ssc
from src.core_sp.value_gate import apply_value_gate
from src.core_sp.generators import generate_semantic_embeddings
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def generate_grid_layout(n_items: int = 64) -> np.ndarray:
    """Generate 8x8 grid layout."""
    grid_size = int(np.sqrt(n_items))
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            coords.append([float(j), float(i)])
    return np.array(coords, dtype=np.float64)


def run_sp50_lambda_tradeoff_grid_synth(
    n_trials: int = 1000,
    seed: int = 700,
    lambda_values: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    out_dir: Path = Path("outputs_sp/sp50_lambda_tradeoff_grid_synth"),
) -> None:
    """Test SSC-SP tradeoff with grid layout (synthetic embeddings)."""
    
    rng = np.random.default_rng(seed)
    
    # Parameters
    n_items = 64
    d_embedding = 128
    
    # Generate grid layout
    base_coords = generate_grid_layout(n_items)
    layout_type = "grid"
    
    print(f"  Grid layout: {base_coords.shape}")
    print(f"  Embedding dimension: {d_embedding}")
    
    records = []
    
    for lam in lambda_values:
        print(f"  Processing λ={lam}...")
        for trial in range(n_trials):
            # Generate synthetic embeddings
            trial_seed = seed + trial
            trial_rng = np.random.default_rng(trial_seed)
            sem = generate_semantic_embeddings(n_items, d_embedding, trial_rng)
            
            # Apply value gate
            coords_value = apply_value_gate(
                base_coords, sem, lam, seed=trial_seed, radius=1.0
            )
            
            # Compute SP and SSC
            sp_val = compute_sp_total(base_coords, coords_value, layout_type=layout_type)
            ssc_val = compute_ssc(sem, coords_value)
            
            records.append({
                "lambda": lam,
                "trial": trial,
                "sp": sp_val,
                "ssc": ssc_val
            })
    
    # Compute summary statistics per λ
    summary_rows = []
    
    for lam in lambda_values:
        lam_records = [r for r in records if r["lambda"] == lam]
        
        sp_values = [r["sp"] for r in lam_records]
        ssc_values = [r["ssc"] for r in lam_records]
        
        sp_stats = compute_statistics(sp_values)
        ssc_stats = compute_statistics(ssc_values)
        
        summary_rows.append({
            "lambda": lam,
            "n": sp_stats["n"],
            "sp_mean": sp_stats["mean"],
            "sp_std": sp_stats["std"],
            "sp_ci_low": sp_stats["ci_low"],
            "sp_ci_high": sp_stats["ci_high"],
            "ssc_mean": ssc_stats["mean"],
            "ssc_std": ssc_stats["std"],
            "ssc_ci_low": ssc_stats["ci_low"],
            "ssc_ci_high": ssc_stats["ci_high"]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    save_experiment_results(
        experiment_id="sp50_lambda_tradeoff_grid_synth",
        version="v2.1.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "n_items": n_items,
            "d_embedding": d_embedding,
            "lambda_values": list(lambda_values),
            "layout_type": layout_type
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp50_lambda_tradeoff_grid_synth()
