"""SP-01: Full Destruction Test.
Instrument Validation: Random relayout (complete topology destruction)
should drastically reduce SP, confirming SP's topology-sensitivity.
Expected Results:
    - SP after destruction << 1 (typically ~0.5)
    - Confirms SP detects topological disruption
Author: HIDEKI
Date: 2025-11
License: MIT
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.topology_ops import random_relayout
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp01_full_destruction(
    n_trials: int = 1000,
    seed: int = 123,
    out_dir: Path = Path("outputs_sp/sp01_full_destruction"),
) -> None:
    """
    Test SP under complete topological destruction.
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials
    seed : int, default=123
        Random seed
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    layout = "grid"
    
    base_coords = make_grid_layout(n_side=6)
    
    records = []
    
    for trial in range(n_trials):
        # Random relayout (complete destruction)
        coords_destroyed = random_relayout(base_coords, rng=rng)
        sp = compute_sp_total(base_coords, coords_destroyed, layout_type=layout)
        
        records.append({
            "trial": trial,
            "sp": sp
        })
    
    # Compute summary statistics
    sp_values = [r["sp"] for r in records]
    stats = compute_statistics(sp_values)
    
    summary_df = pd.DataFrame([{
        "layout": layout,
        "n": stats["n"],
        "sp_mean": stats["mean"],
        "sp_std": stats["std"],
        "sp_ci_low": stats["ci_low"],
        "sp_ci_high": stats["ci_high"]
    }])
    
    # Save results
    save_experiment_results(
        experiment_id="sp01_full_destruction",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "layout_type": layout
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp01_full_destruction()
