"""SP-40: Dimension/N Robustness Test.

Robustness Validation: Verify that topological dominance pattern
(topology disruption >> metric distortion effect) holds across
different numbers of items N.

Expected Results:
    - Topology disruption: SP << 1 for all N
    - Metric distortion: SP ≈ 1 for all N
    - Pattern consistent across scales

Author: HIDEKI
Date: 2025-11
License: MIT
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.metric_ops import shear_2d
from src.core_sp.topology_ops import permute_coords
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp40_dimN_sp_robustness(
    n_trials: int = 200,
    seed: int = 700,
    n_items_list: tuple = (16, 36, 64, 100),
    out_dir: Path = Path("outputs_sp/sp40_dimN_sp_robustness"),
) -> None:
    """Test robustness of topological dominance across different N.
    
    Parameters
    ----------
    n_trials : int, default=200
        Number of trials per N × family
    seed : int, default=700
        Random seed for reproducibility
    n_items_list : tuple
        Number of items to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    
    records = []
    
    for n_items in n_items_list:
        # Approximate grid: sqrt(n_items) × sqrt(n_items)
        n_side = int(np.sqrt(n_items))
        base_coords = make_grid_layout(n_side=n_side)
        layout_type = "grid"
        
        for trial in range(n_trials):
            # Topology disruption (permute with p=0.5)
            coords_perm = permute_coords(base_coords, rng=rng, p=0.5)
            sp_topo = compute_sp_total(base_coords, coords_perm, layout_type=layout_type)
            
            # Metric distortion (shear with k=0.7)
            coords_shear = shear_2d(base_coords, k=0.7)
            sp_metric = compute_sp_total(base_coords, coords_shear, layout_type=layout_type)
            
            records.append({
                "n_items": n_items,
                "trial": trial,
                "sp_topology": sp_topo,
                "sp_metric": sp_metric
            })
    
    # Compute summary statistics per n_items
    summary_rows = []
    
    for n_items in n_items_list:
        n_records = [r for r in records if r["n_items"] == n_items]
        
        sp_topo_values = [r["sp_topology"] for r in n_records]
        sp_metric_values = [r["sp_metric"] for r in n_records]
        
        topo_stats = compute_statistics(sp_topo_values)
        metric_stats = compute_statistics(sp_metric_values)
        
        summary_rows.append({
            "n_items": n_items,
            "n": topo_stats["n"],
            "sp_topology_mean": topo_stats["mean"],
            "sp_topology_std": topo_stats["std"],
            "sp_topology_ci_low": topo_stats["ci_low"],
            "sp_topology_ci_high": topo_stats["ci_high"],
            "sp_metric_mean": metric_stats["mean"],
            "sp_metric_std": metric_stats["std"],
            "sp_metric_ci_low": metric_stats["ci_low"],
            "sp_metric_ci_high": metric_stats["ci_high"]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    save_experiment_results(
        experiment_id="sp40_dimN_sp_robustness",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "n_items_list": list(n_items_list)
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp40_dimN_sp_robustness()
