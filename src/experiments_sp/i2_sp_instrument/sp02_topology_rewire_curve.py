"""SP-02: Topology Rewire Curve.
Instrument Validation: Coordinate permutation with increasing probability p
should show monotonic decrease in SP_adj, demonstrating sensitivity
to topological disruption strength.
Expected Results:
    - p=0: SP_adj ≈ 1 (no disruption)
    - p↑: SP_adj↓ (monotonic decrease)
    - p=1: SP_adj << 1 (complete disruption)
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
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp02_topology_rewire_curve(
    n_trials: int = 1000,
    seed: int = 77,
    p_values: tuple = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
    out_dir: Path = Path("outputs_sp/sp02_topology_rewire_curve"),
) -> None:
    """
    SP response to coordinate permutation probability.
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per p value
    seed : int, default=77
        Random seed
    p_values : tuple, default=(0.0, 0.1, 0.3, 0.5, 0.7, 1.0)
        Permutation probabilities to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    layout = "grid"
    
    base_coords = make_grid_layout(n_side=6)
    
    records = []
    
    for p in p_values:
        for trial in range(n_trials):
            # Permute coordinates (topology disruption)
            coords_permuted = permute_coords(base_coords, rng=rng, p=p)
            sp_adj = compute_sp_total(base_coords, coords_permuted, layout_type=layout)
            
            records.append({
                "p": p,
                "trial": trial,
                "sp_adj": sp_adj
            })
    
    # Compute summary statistics per p
    summary_rows = []
    
    for p in p_values:
        p_records = [r for r in records if r["p"] == p]
        sp_adj_values = [r["sp_adj"] for r in p_records]
        stats = compute_statistics(sp_adj_values)
        
        summary_rows.append({
            "p": p,
            "n": stats["n"],
            "sp_adj_mean": stats["mean"],
            "sp_adj_std": stats["std"],
            "sp_adj_ci_low": stats["ci_low"],
            "sp_adj_ci_high": stats["ci_high"]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save results
    save_experiment_results(
        experiment_id="sp02_topology_rewire_curve",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "layout": layout,
            "p_values": list(p_values)
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp02_topology_rewire_curve()
