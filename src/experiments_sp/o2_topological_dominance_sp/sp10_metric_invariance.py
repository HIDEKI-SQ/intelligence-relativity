"""SP-10: Metric Invariance Test.
O-2: Topological Dominance - Verify that SP is preserved under
metric transformations (rotation, scaling, shearing) but sensitive
to topological disruptions.
Expected Results:
    - Rotation: SP ≈ constant (metric invariant)
    - Scaling: SP ≈ constant
    - Shearing: SP slightly reduced but stable
Author: HIDEKI
Date: 2025-11
License: MIT
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.metric_ops import rotate_2d, scale_2d, shear_2d
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp10_metric_invariance(
    n_trials: int = 1000,
    seed: int = 200,
    out_dir: Path = Path("outputs_sp/sp10_metric_invariance"),
) -> None:
    """Test SP invariance under various metric transformations."""
    rng = np.random.default_rng(seed)
    layout = "grid"
    
    # Test conditions
    rotations_deg = [0.0, 30.0, 60.0, 90.0, 120.0, 180.0]
    scales = [(0.5, 2.0), (2.0, 0.5), (1.0, 1.0)]
    shears = [0.0, 0.3, 0.7, 1.0]
    
    base_coords = make_grid_layout(n_side=6)
    
    records = []
    
    for trial in range(n_trials):
        # Rotation tests
        for theta_deg in rotations_deg:
            theta_rad = np.deg2rad(theta_deg)
            coords_rot = rotate_2d(base_coords, theta_rad=theta_rad)
            sp = compute_sp_total(base_coords, coords_rot, layout_type=layout)
            
            records.append({
                "transform": "rotation",
                "theta_deg": theta_deg,
                "sx": None,
                "sy": None,
                "k": None,
                "trial": trial,
                "sp": sp
            })
        
        # Scale tests
        for sx, sy in scales:
            coords_scale = scale_2d(base_coords, sx=sx, sy=sy)
            sp = compute_sp_total(base_coords, coords_scale, layout_type=layout)
            
            records.append({
                "transform": "scale",
                "theta_deg": None,
                "sx": sx,
                "sy": sy,
                "k": None,
                "trial": trial,
                "sp": sp
            })
        
        # Shear tests
        for k in shears:
            coords_shear = shear_2d(base_coords, k=k)
            sp = compute_sp_total(base_coords, coords_shear, layout_type=layout)
            
            records.append({
                "transform": "shear",
                "theta_deg": None,
                "sx": None,
                "sy": None,
                "k": k,
                "trial": trial,
                "sp": sp
            })
    
    # Compute summary statistics
    summary_rows = []
    
    # Rotation
    for theta in rotations_deg:
        theta_records = [r for r in records 
                        if r["transform"] == "rotation" and r["theta_deg"] == theta]
        sp_values = [r["sp"] for r in theta_records]
        stats = compute_statistics(sp_values)
        
        summary_rows.append({
            "transform": "rotation",
            "theta_deg": theta,
            "sx": None,
            "sy": None,
            "k": None,
            "n": stats["n"],
            "sp_mean": stats["mean"],
            "sp_std": stats["std"],
            "sp_ci_low": stats["ci_low"],
            "sp_ci_high": stats["ci_high"]
        })
    
    # Scale
    for sx, sy in scales:
        scale_records = [r for r in records 
                        if r["transform"] == "scale" and r["sx"] == sx and r["sy"] == sy]
        sp_values = [r["sp"] for r in scale_records]
        stats = compute_statistics(sp_values)
        
        summary_rows.append({
            "transform": "scale",
            "theta_deg": None,
            "sx": sx,
            "sy": sy,
            "k": None,
            "n": stats["n"],
            "sp_mean": stats["mean"],
            "sp_std": stats["std"],
            "sp_ci_low": stats["ci_low"],
            "sp_ci_high": stats["ci_high"]
        })
    
    # Shear
    for k in shears:
        shear_records = [r for r in records 
                        if r["transform"] == "shear" and r["k"] == k]
        sp_values = [r["sp"] for r in shear_records]
        stats = compute_statistics(sp_values)
        
        summary_rows.append({
            "transform": "shear",
            "theta_deg": None,
            "sx": None,
            "sy": None,
            "k": k,
            "n": stats["n"],
            "sp_mean": stats["mean"],
            "sp_std": stats["std"],
            "sp_ci_low": stats["ci_low"],
            "sp_ci_high": stats["ci_high"]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    save_experiment_results(
        experiment_id="sp10_metric_invariance",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "layout": layout,
            "rotations_deg": rotations_deg,
            "scales": scales,
            "shears": shears
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp10_metric_invariance()
