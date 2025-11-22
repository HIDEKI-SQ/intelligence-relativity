"""SP-12: Topology vs Metric Comparison.
O-2: Topological Dominance - Direct comparison of SP response to
topological disruptions vs metric distortions at matched intensities.
Expected Results:
    - Topology family: SP sharply decreases
    - Metric family: SP remains stable
    - Clear dominance of topology
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


def run_sp12_topology_vs_metric(
    n_trials: int = 1000,
    seed: int = 202,
    levels: tuple = (0.0, 0.3, 0.7),
    out_dir: Path = Path("outputs_sp/sp12_topology_vs_metric"),
) -> None:
    """Compare topology vs metric effects on SP."""
    rng = np.random.default_rng(seed)
    layout = "grid"
    
    base_coords = make_grid_layout(n_side=6)
    
    records = []
    
    for level in levels:
        for trial in range(n_trials):
            # Topology disruption (permute)
            coords_topo = permute_coords(base_coords, rng=rng, p=level)
            sp_topo = compute_sp_total(base_coords, coords_topo, layout_type=layout)
            
            records.append({
                "family": "topology",
                "level": level,
                "trial": trial,
                "sp": sp_topo
            })
            
            # Metric distortion (shear)
            coords_metric = shear_2d(base_coords, k=level)
            sp_metric = compute_sp_total(base_coords, coords_metric, layout_type=layout)
            
            records.append({
                "family": "metric",
                "level": level,
                "trial": trial,
                "sp": sp_metric
            })
    
    # Compute summary statistics
    summary_rows = []
    
    for family in ["topology", "metric"]:
        for level in levels:
            family_records = [r for r in records 
                            if r["family"] == family and r["level"] == level]
            sp_values = [r["sp"] for r in family_records]
            stats = compute_statistics(sp_values)
            
            summary_rows.append({
                "family": family,
                "level": level,
                "n": stats["n"],
                "sp_mean": stats["mean"],
                "sp_std": stats["std"],
                "sp_ci_low": stats["ci_low"],
                "sp_ci_high": stats["ci_high"]
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    save_experiment_results(
        experiment_id="sp12_topology_vs_metric",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "layout": layout,
            "families": ["topology", "metric"],
            "levels": list(levels)
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp12_topology_vs_metric()
