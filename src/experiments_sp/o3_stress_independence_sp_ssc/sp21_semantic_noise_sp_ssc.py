"""SP-21: Semantic Noise vs SP & SSC.

O-3 Stress Tolerance: Verify that semantic noise may affect SSC while
SP remains ~1 (independence along semantic axis, coords fixed).

Expected Results:
    - SP ≈ 1.0 regardless of σ_sem (coords unchanged)
    - SSC behavior depends on noise (may remain ~0 or fluctuate)

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
from src.core_sp.generators import generate_semantic_embeddings, add_semantic_noise
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp21_semantic_noise_sp_ssc(
    n_trials: int = 1000,
    seed: int = 501,
    n_items: int = 64,
    dim: int = 128,
    sigma_sem_values: tuple = (0.0, 0.1, 0.3, 0.5, 0.7),
    out_dir: Path = Path("outputs_sp/sp21_semantic_noise_sp_ssc"),
) -> None:
    """Test independence of SP and SSC under semantic noise.
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per sigma value
    seed : int, default=501
        Random seed for reproducibility
    n_items : int, default=64
        Number of items
    dim : int, default=128
        Embedding dimension
    sigma_sem_values : tuple
        Semantic noise levels to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"
    
    records = []
    
    for sigma_sem in sigma_sem_values:
        for trial in range(n_trials):
            # Generate semantic embeddings
            sem_clean = generate_semantic_embeddings(n_items, dim, rng)
            sem_noisy = add_semantic_noise(sem_clean, rng=rng, sigma=sigma_sem)
            
            # SP with identical coordinates (should remain constant)
            sp_val = compute_sp_total(base_coords, base_coords, layout_type=layout_type)
            
            # SSC with noisy semantics
            ssc_val = compute_ssc(sem_noisy, base_coords)
            
            records.append({
                "sigma_sem": sigma_sem,
                "trial": trial,
                "sp": sp_val,
                "ssc": ssc_val
            })
    
    # Compute summary statistics per sigma_sem
    summary_rows = []
    
    for sigma_sem in sigma_sem_values:
        sigma_records = [r for r in records if r["sigma_sem"] == sigma_sem]
        
        sp_values = [r["sp"] for r in sigma_records]
        ssc_values = [r["ssc"] for r in sigma_records]
        
        sp_stats = compute_statistics(sp_values)
        ssc_stats = compute_statistics(ssc_values)
        
        summary_rows.append({
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
        experiment_id="sp21_semantic_noise_sp_ssc",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "n_items": n_items,
            "dim": dim,
            "sigma_sem_values": list(sigma_sem_values)
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp21_semantic_noise_sp_ssc()
