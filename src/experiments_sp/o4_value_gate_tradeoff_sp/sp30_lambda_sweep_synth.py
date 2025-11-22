"""SP-30: λ Sweep (Synthetic Embeddings).

O-4 Value-Gated Coupling: Verify that increasing λ produces the expected
tradeoff: SSC ↑ (semantic-spatial coupling increases) while SP ↓
(structural preservation decreases due to semantic reordering).

Expected Results:
    - λ=0: SSC ≈ 0, SP ≈ 1 (natural orthogonality, structure preserved)
    - λ↑: SSC ↑, SP ↓ (value-gated coupling, tradeoff emerges)
    - Linear regime followed by saturation

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
from src.core_sp.generators import generate_semantic_embeddings
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp30_lambda_sweep_synth(
    n_trials: int = 1000,
    seed: int = 600,
    n_items: int = 64,
    dim: int = 128,
    lambda_values: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    out_dir: Path = Path("outputs_sp/sp30_lambda_sweep_synth"),
) -> None:
    """Test SSC-SP tradeoff with value gate parameter λ (synthetic embeddings).
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per λ value
    seed : int, default=600
        Random seed for reproducibility
    n_items : int, default=64
        Number of items (must match grid layout: 8x8=64)
    dim : int, default=128
        Embedding dimension
    lambda_values : tuple
        Value gate parameters to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"
    
    records = []
    
    for lam in lambda_values:
        for trial in range(n_trials):
            # Generate semantic embeddings
            sem = generate_semantic_embeddings(n_items, dim, rng)
            
            # Apply value gate to create λ-modulated coordinates
            trial_seed = seed + trial
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
        experiment_id="sp30_lambda_sweep_synth",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "n_items": n_items,
            "dim": dim,
            "lambda_values": list(lambda_values)
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp30_lambda_sweep_synth()
