"""SP-03: Layout Robustness Test.
Instrument Validation: SP should behave consistently across different
spatial layouts (grid, line, circle, random), showing robustness of
the measurement.
Expected Results:
    - Identity: SP high for all layouts
    - Destruction: SP reduced for all layouts
    - Pattern consistent across layouts
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
from src.core.generators import generate_spatial_coords
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp03_layout_robustness(
    n_trials: int = 1000,
    seed: int = 55,
    out_dir: Path = Path("outputs_sp/sp03_layout_robustness"),
) -> None:
    """
    Test SP consistency across layouts.
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials for destruction case
    seed : int, default=55
        Random seed
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    layouts = ["grid", "line", "circle", "random"]
    n_items = 36
    dim = 2
    
    records = []
    
    for layout in layouts:
        # Generate base coordinates
        base_coords = generate_spatial_coords(
            n_items=n_items,
            dim=dim,
            layout=layout,
            seed=seed
        )
        
        # Identity case (1 trial)
        sp_identity = compute_sp_total(base_coords, base_coords, layout_type=layout)
        records.append({
            "layout": layout,
            "case": "identity",
            "trial": 0,
            "sp": sp_identity
        })
        
        # Destruction case (n_trials)
        for trial in range(n_trials):
            coords_destroyed = random_relayout(base_coords, rng=rng)
            sp_destroyed = compute_sp_total(base_coords, coords_destroyed, layout_type=layout)
            
            records.append({
                "layout": layout,
                "case": "destruction",
                "trial": trial,
                "sp": sp_destroyed
            })
    
    # Compute summary statistics per (layout, case)
    summary_rows = []
    
    for layout in layouts:
        for case in ["identity", "destruction"]:
            case_records = [r for r in records 
                          if r["layout"] == layout and r["case"] == case]
            sp_values = [r["sp"] for r in case_records]
            stats = compute_statistics(sp_values)
            
            summary_rows.append({
                "layout": layout,
                "case": case,
                "n": stats["n"],
                "sp_mean": stats["mean"],
                "sp_std": stats["std"],
                "sp_ci_low": stats["ci_low"],
                "sp_ci_high": stats["ci_high"]
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save results
    save_experiment_results(
        experiment_id="sp03_layout_robustness",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "layouts": layouts,
            "n_items": n_items,
            "dim": dim
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp03_layout_robustness()
