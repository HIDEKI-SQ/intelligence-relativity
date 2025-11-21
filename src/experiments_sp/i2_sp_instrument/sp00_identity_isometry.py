"""SP-00: Identity & Isometry Test.

I-2 Instrument Validation: Verify that isometric transformations
(rotation, uniform scaling, translation) preserve SP ≈ 1.

Expected Results:
    - Rotation: SP ≈ 1.0 across all angles
    - Uniform scaling: SP ≈ 1.0 across all scales
    - Identity: SP = 1.0 (perfect preservation)

Author: HIDEKI
Date: 2025-11
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.metric_ops import rotate_2d, scale_2d


def make_grid_layout(n_side: int = 8, span: float = 2.0) -> np.ndarray:
    """
    Generate a simple 2D grid layout.
    
    Parameters
    ----------
    n_side : int, default=8
        Number of points per side
    span : float, default=2.0
        Grid span in [-span/2, span/2]
    
    Returns
    -------
    coords : ndarray, shape (n_side^2, 2)
        Grid coordinates
    """
    xs = np.linspace(-span / 2.0, span / 2.0, n_side)
    ys = np.linspace(-span / 2.0, span / 2.0, n_side)
    xv, yv = np.meshgrid(xs, ys)
    coords = np.stack([xv.ravel(), yv.ravel()], axis=1)
    return coords


def run_sp00_identity_isometry(
    n_trials: int = 100,
    seed: int = 42,
    out_dir: Path = Path("outputs_sp/i2_sp_instrument/sp00_identity_isometry"),
) -> None:
    """
    I-2: SP-00 Identity & Isometry test.
    
    For each trial, apply a set of isometries to a base grid layout and
    verify that SP ≈ 1.
    
    Parameters
    ----------
    n_trials : int, default=100
        Number of trials per transformation
    seed : int, default=42
        Random seed for reproducibility
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_coords = make_grid_layout(n_side=8, span=2.0)
    layout_type = "grid"

    angles_deg = [0.0, 30.0, 60.0, 90.0, 120.0, 180.0]
    scales = [(1.0, 1.0), (2.0, 2.0), (0.5, 0.5)]  # Uniform isotropic scales

    records = []

    for trial in range(n_trials):
        # Test rotations
        for theta_deg in angles_deg:
            theta_rad = np.deg2rad(theta_deg)
            coords_rot = rotate_2d(base_coords, theta_rad, center=(0.0, 0.0))
            sp_val = compute_sp_total(base_coords, coords_rot, layout_type=layout_type)
            records.append(
                {
                    "trial": trial,
                    "transform": "rotation",
                    "theta_deg": theta_deg,
                    "sp": sp_val,
                }
            )

        # Test uniform scaling
        for sx, sy in scales:
            coords_scaled = scale_2d(base_coords, sx=sx, sy=sy, center=(0.0, 0.0))
            sp_val = compute_sp_total(base_coords, coords_scaled, layout_type=layout_type)
            records.append(
                {
                    "trial": trial,
                    "transform": "scale",
                    "sx": sx,
                    "sy": sy,
                    "sp": sp_val,
                }
            )

    # Save raw records
    out_path = out_dir / "sp00_identity_isometry_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "layout_type": layout_type,
                    "angles_deg": angles_deg,
                    "scales": scales,
                },
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✅ Saved raw records to {out_path}")
    print(f"   Total records: {len(records)}")


if __name__ == "__main__":
    run_sp00_identity_isometry()
