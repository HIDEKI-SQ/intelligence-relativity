"""SP-20: Coordinate Noise vs SP & SSC.

O-3 Stress Tolerance: Verify that coordinate noise lowers SP while
SSC remains ~0 under λ=0 (independence along coordinate axis).

Expected Results:
    - σ_coord ↑ → SP ↓ (coordinate structure degraded)
    - SSC ≈ 0 regardless of σ_coord (λ=0, natural orthogonality)

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
from src.core_sp.generators import generate_semantic_embeddings
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp20_coord_noise_sp_ssc(
    n_trials: int = 1000,
    seed: int = 500,
    n_items: int = 64,
    dim: int = 128,
    sigma_coord_values: tuple = (0.0, 0.1, 0.3, 0.5, 0.7),
    out_dir: Path = Path("outputs_sp/sp20_coord_noise_sp_ssc"),
) -> None:
    """Test independence of SP and SSC under coordinate noise.
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per sigma value
    seed : int, default=500
        Random seed for reproducibility
    n_items : int, default=64
        Number of items (must match grid layout: 8x8=64)
    dim : int, default=128
        Embedding dimension
    sigma_coord_values : tuple
        Coordinate noise levels to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"
    
    records = []
    
    for sigma_coord in sigma_coord_values:
        for trial in range(n_trials):
            # Generate semantic embeddings (same for all trials at λ=0)
            sem = generate_semantic_embeddings(n_items, dim, rng)
            
            # Add coordinate noise
            coords_noisy = add_coord_noise(base_coords, rng=rng, sigma=sigma_coord)
            
            # Compute SP and SSC
            sp_val = compute_sp_total(base_coords, coords_noisy, layout_type=layout_type)
            ssc_val = compute_ssc(sem, coords_noisy)  # λ=0 natural state
            
            records.append({
                "sigma_coord": sigma_coord,
                "trial": trial,
                "sp": sp_val,
                "ssc": ssc_val
            })
    
    # Compute summary statistics per sigma_coord
    summary_rows = []
    
    for sigma_coord in sigma_coord_values:
        sigma_records = [r for r in records if r["sigma_coord"] == sigma_coord]
        
        sp_values = [r["sp"] for r in sigma_records]
        ssc_values = [r["ssc"] for r in sigma_records]
        
        sp_stats = compute_statistics(sp_values)
        ssc_stats = compute_statistics(ssc_values)
        
        summary_rows.append({
            "sigma_coord": sigma_coord,
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
        experiment_id="sp20_coord_noise_sp_ssc",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "n_items": n_items,
            "dim": dim,
            "sigma_coord_values": list(sigma_coord_values)
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp20_coord_noise_sp_ssc()
