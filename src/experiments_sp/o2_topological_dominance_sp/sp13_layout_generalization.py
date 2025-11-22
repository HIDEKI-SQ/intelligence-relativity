"""SP-13: Layout Generalization Test.
O-2: Topological Dominance - Verify that topology vs metric pattern
generalizes across different spatial layouts.
Expected Results:
    - Pattern holds for all layouts
    - Metric: SP stable
    - Topology: SP decreases
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
from src.core.generators import generate_spatial_coords
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp13_layout_generalization(
    n_trials: int = 1000,
    seed: int = 203,
    out_dir: Path = Path("outputs_sp/sp13_layout_generalization"),
) -> None:
    """Test topology vs metric pattern across layouts."""
    rng = np.random.default_rng(seed)
    
    layouts = ["grid", "line", "circle", "random"]
    n_items = 36
    test_level = 0.7  # High disruption/distortion level
    
    records = []
    
    for layout in layouts:
        base_coords = generate_spatial_coords(
            n_items=n_items,
            layout=layout,
            seed=seed
        )
        
        for trial in range(n_trials):
            # Metric distortion
            coords_metric = shear_2d(base_coords, k=test_level)
            sp_metric = compute_sp_total(base_coords, coords_metric, layout_type=layout)
            
            records.append({
                "layout": layout,
                "family": "metric",
                "level": test_level,
                "trial": trial,
                "sp": sp_metric
            })
            
            # Topology disruption
            coords_topo = permute_coords(base_coords, rng=rng, p=test_level)
            sp_topo = compute_sp_total(base_coords, coords_topo, layout_type=layout)
            
            records.append({
                "layout": layout,
                "family": "topology",
                "level": test_level,
                "trial": trial,
                "sp": sp_topo
            })
    
    # Compute summary statistics
    summary_rows = []
    
    for layout in layouts:
        for family in ["metric", "topology"]:
            family_records = [r for r in records 
                            if r["layout"] == layout and r["family"] == family]
            sp_values = [r["sp"] for r in family_records]
            stats = compute_statistics(sp_values)
            
            summary_rows.append({
                "layout": layout,
                "family": family,
                "level": test_level,
                "n": stats["n"],
                "sp_mean": stats["mean"],
                "sp_std": stats["std"],
                "sp_ci_low": stats["ci_low"],
                "sp_ci_high": stats["ci_high"]
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    save_experiment_results(
        experiment_id="sp13_layout_generalization",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "layouts": layouts,
            "n_items": n_items,
            "families": ["metric", "topology"],
            "test_level": test_level
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp13_layout_generalization()
