"""SP-41: Layout × Topology Robustness Test.

Robustness Validation: Verify that topology disruption effects
generalize across multiple layouts (grid, line, circle, random).

Expected Results:
    - p=0.0: SP ≈ 1 for all layouts
    - p↑: SP ↓ for all layouts
    - Pattern consistent across layout types

Author: HIDEKI
Date: 2025-11
License: MIT
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.topology_ops import permute_coords
from src.experiments_sp.i2_sp_instrument.sp03_layout_robustness import (
    make_line_layout,
    make_circle_layout,
    make_random_layout,
)
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp41_layout_topology_sp_robustness(
    n_trials: int = 200,
    seed: int = 701,
    p_values: tuple = (0.0, 0.3, 0.7),
    out_dir: Path = Path("outputs_sp/sp41_layout_topology_sp_robustness"),
) -> None:
    """Test robustness of topology disruption across different layouts.
    
    Parameters
    ----------
    n_trials : int, default=200
        Number of trials per layout × p
    seed : int, default=701
        Random seed for reproducibility
    p_values : tuple
        Permutation proportions to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    
    layouts = {
        "grid": make_grid_layout(n_side=8),
        "line": make_line_layout(),
        "circle": make_circle_layout(),
        "random": make_random_layout(64, rng=rng),
    }
    
    records = []
    
    for layout_name, coords in layouts.items():
        for p in p_values:
            for trial in range(n_trials):
                coords_perm = permute_coords(coords, rng=rng, p=p)
                sp_val = compute_sp_total(coords, coords_perm, layout_type=layout_name)
                
                records.append({
                    "layout": layout_name,
                    "p": p,
                    "trial": trial,
                    "sp": sp_val
                })
    
    # Compute summary statistics per (layout, p) combination
    summary_rows = []
    
    for layout_name in layouts.keys():
        for p in p_values:
            combo_records = [r for r in records 
                           if r["layout"] == layout_name and r["p"] == p]
            
            sp_values = [r["sp"] for r in combo_records]
            stats = compute_statistics(sp_values)
            
            summary_rows.append({
                "layout": layout_name,
                "p": p,
                "n": stats["n"],
                "sp_mean": stats["mean"],
                "sp_std": stats["std"],
                "sp_ci_low": stats["ci_low"],
                "sp_ci_high": stats["ci_high"]
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    save_experiment_results(
        experiment_id="sp41_layout_topology_sp_robustness",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "p_values": list(p_values)
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp41_layout_topology_sp_robustness()
