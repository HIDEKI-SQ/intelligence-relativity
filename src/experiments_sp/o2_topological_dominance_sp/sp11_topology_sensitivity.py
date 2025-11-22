"""SP-11: Topology Sensitivity Test.
O-2: Topological Dominance - Measure SP sensitivity to
increasing levels of topological disruption (coordinate permutation).
Expected Results:
    - p=0: SP high (no disruption)
    - p↑: SP↓ (monotonic decrease)
    - Demonstrates topology-sensitivity
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


def run_sp11_topology_sensitivity(
    n_trials: int = 1000,
    seed: int = 201,
    p_values: tuple = (0.0, 0.1, 0.3, 0.5, 0.7),
    out_dir: Path = Path("outputs_sp/sp11_topology_sensitivity"),
) -> None:
    """Test SP sensitivity to topological disruption strength."""
    rng = np.random.default_rng(seed)
    layout = "grid"
    
    base_coords = make_grid_layout(n_side=6)
    
    records = []
    
    for p in p_values:
        for trial in range(n_trials):
            coords_permuted = permute_coords(base_coords, rng=rng, p=p)
            sp = compute_sp_total(base_coords, coords_permuted, layout_type=layout)
            
            records.append({
                "p": p,
                "trial": trial,
                "sp": sp
            })
    
    # Compute summary statistics
    summary_rows = []
    
    for p in p_values:
        p_records = [r for r in records if r["p"] == p]
        sp_values = [r["sp"] for r in p_records]
        stats = compute_statistics(sp_values)
        
        summary_rows.append({
            "p": p,
            "n": stats["n"],
            "sp_mean": stats["mean"],
            "sp_std": stats["std"],
            "sp_ci_low": stats["ci_low"],
            "sp_ci_high": stats["ci_high"]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    save_experiment_results(
        experiment_id="sp11_topology_sensitivity",
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
    run_sp11_topology_sensitivity()
