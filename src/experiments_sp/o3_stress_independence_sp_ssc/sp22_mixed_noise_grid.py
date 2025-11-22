"""SP-22: Mixed Noise Grid (2D Parameter Sweep).

O-3 Stress Tolerance: 2D grid exploration of coordinate noise × semantic noise
to demonstrate independence: SP depends only on σ_coord, SSC ≈ 0 at λ=0.

Expected Results:
    - SP = f(σ_coord) only (independent of σ_sem)
    - SSC ≈ 0 for all (σ_coord, σ_sem) combinations (λ=0)

Author: HIDEKI
Date: 2025-11
License: MIT
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.metric_ops import add_coord_noise
from src.core_sp.ssc_wrapper import compute_ssc
from src.core_sp.generators import generate_semantic_embeddings, add_semantic_noise
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp22_mixed_noise_grid(
    n_trials: int = 1000,
    seed: int = 502,
    n_items: int = 64,
    dim: int = 128,
    sigma_coord_values: tuple = (0.0, 0.1, 0.3, 0.5),
    sigma_sem_values: tuple = (0.0, 0.1, 0.3, 0.5),
    out_dir: Path = Path("outputs_sp/sp22_mixed_noise_grid"),
) -> None:
    """Test independence with 2D grid of coordinate and semantic noise.
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per (σ_coord, σ_sem) pair
    seed : int, default=502
        Random seed for reproducibility
    n_items : int, default=64
        Number of items
    dim : int, default=128
        Embedding dimension
    sigma_coord_values : tuple
        Coordinate noise levels
    sigma_sem_values : tuple
        Semantic noise levels
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"
    
    records = []
    
    for sigma_coord in sigma_coord_values:
        for sigma_sem in sigma_sem_values:
            for trial in range(n_trials):
                # Generate and add noise to both axes
                sem = generate_semantic_embeddings(n_items, dim, rng)
                sem_noisy = add_semantic_noise(sem, rng=rng, sigma=sigma_sem)
                coords_noisy = add_coord_noise(base_coords, rng=rng, sigma=sigma_coord)
                
                # Compute SP and SSC
                sp_val = compute_sp_total(base_coords, coords_noisy, layout_type=layout_type)
                ssc_val = compute_ssc(sem_noisy, coords_noisy)
                
                records.append({
                    "sigma_coord": sigma_coord,
                    "sigma_sem": sigma_sem,
                    "trial": trial,
                    "sp": sp_val,
                    "ssc": ssc_val
                })
    
    # Compute summary statistics per (sigma_coord, sigma_sem) pair
    summary_rows = []
    
    for sigma_coord in sigma_coord_values:
        for sigma_sem in sigma_sem_values:
            pair_records = [r for r in records 
                           if r["sigma_coord"] == sigma_coord 
                           and r["sigma_sem"] == sigma_sem]
            
            sp_values = [r["sp"] for r in pair_records]
            ssc_values = [r["ssc"] for r in pair_records]
            
            sp_stats = compute_statistics(sp_values)
            ssc_stats = compute_statistics(ssc_values)
            
            summary_rows.append({
                "sigma_coord": sigma_coord,
                "sigma_sem": sigma_sem,
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
        experiment_id="sp22_mixed_noise_grid",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "n_items": n_items,
            "dim": dim,
            "sigma_coord_values": list(sigma_coord_values),
            "sigma_sem_values": list(sigma_sem_values)
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp22_mixed_noise_grid()
