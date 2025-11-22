"""SP-00: Identity & Isometry Test.
Instrument Validation: Verify that isometric transformations
(rotation, uniform scaling) preserve SP, demonstrating that
SP measures topology, not geometry.
Expected Results:
    - Rotation: SP invariant across angles
    - Uniform scaling: SP invariant
    - Identity: SP ≈ 1
Author: HIDEKI
Date: 2025-11
License: MIT
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.metric_ops import rotate_2d, scale_2d
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def make_grid_layout(n_side: int = 6) -> np.ndarray:
    """Generate n_side × n_side grid layout."""
    x = np.linspace(0, 1, n_side)
    y = np.linspace(0, 1, n_side)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def run_sp00_identity_isometry(
    n_trials: int = 1,
    seed: int = 42,
    out_dir: Path = Path("outputs_sp/sp00_identity_isometry"),
) -> None:
    """
    Test SP invariance under isometric transformations.
    
    Parameters
    ----------
    n_trials : int, default=1
        Number of trials (typically 1 for deterministic transforms)
    seed : int, default=42
        Random seed
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    
    # Parameters
    layout = "grid"
    rotations_deg = [0.0, 30.0, 60.0, 90.0, 120.0, 180.0]
    scales = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]
    
    base_coords = make_grid_layout(n_side=6)
    
    records = []
    
    for trial in range(n_trials):
        # Rotation tests
        for theta_deg in rotations_deg:
            theta_rad = np.deg2rad(theta_deg)  # 度→ラジアン変換
            coords_rot = rotate_2d(base_coords, theta_rad=theta_rad)
            sp = compute_sp_total(base_coords, coords_rot, layout_type=layout)
            
            records.append({
                "layout": layout,
                "transform": "rotation",
                "theta_deg": theta_deg,
                "sx": None,
                "sy": None,
                "sp": sp
            })
        
        # Scale tests
        for sx, sy in scales:
            coords_scale = scale_2d(base_coords, sx=sx, sy=sy)
            sp = compute_sp_total(base_coords, coords_scale, layout_type=layout)
            
            records.append({
                "layout": layout,
                "transform": "scale",
                "theta_deg": None,
                "sx": sx,
                "sy": sy,
                "sp": sp
            })
    
    # Compute summary statistics
    summary_rows = []
    
    # Group by transform type
    for transform in ["rotation", "scale"]:
        transform_records = [r for r in records if r["transform"] == transform]
        
        if transform == "rotation":
            # Group by theta_deg
            for theta in rotations_deg:
                theta_records = [r for r in transform_records if r["theta_deg"] == theta]
                sp_values = [r["sp"] for r in theta_records]
                stats = compute_statistics(sp_values)
                
                summary_rows.append({
                    "layout": layout,
                    "transform": transform,
                    "theta_deg": theta,
                    "sx": None,
                    "sy": None,
                    "n": stats["n"],
                    "sp_mean": stats["mean"],
                    "sp_std": stats["std"],
                    "sp_ci_low": stats["ci_low"],
                    "sp_ci_high": stats["ci_high"]
                })
        
        elif transform == "scale":
            # Group by (sx, sy)
            for sx, sy in scales:
                scale_records = [r for r in transform_records 
                               if r["sx"] == sx and r["sy"] == sy]
                sp_values = [r["sp"] for r in scale_records]
                stats = compute_statistics(sp_values)
                
                summary_rows.append({
                    "layout": layout,
                    "transform": transform,
                    "theta_deg": None,
                    "sx": sx,
                    "sy": sy,
                    "n": stats["n"],
                    "sp_mean": stats["mean"],
                    "sp_std": stats["std"],
                    "sp_ci_low": stats["ci_low"],
                    "sp_ci_high": stats["ci_high"]
                })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save results
    save_experiment_results(
        experiment_id="sp00_identity_isometry",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "layouts": [layout],
            "rotations_deg": rotations_deg,
            "scales": scales
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp00_identity_isometry()
