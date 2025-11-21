"""SP-10: Metric Invariance Test.

O-2 Topological Dominance: Verify that metric distortions
(rotation, scaling, shear) preserve topology and maintain SP ≈ 1.

Expected Results:
    - Rotation: SP ≈ 1.0 across all angles
    - Anisotropic scaling: SP ≈ 1.0
    - Shear: SP ≈ 1.0 (topology preserved despite metric distortion)

Author: HIDEKI
Date: 2025-11
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.metric_ops import rotate_2d, scale_2d, shear_2d
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout


def run_sp10_metric_invariance(
    n_trials: int = 200,
    seed: int = 101,
    out_dir: Path = Path("outputs_sp/o2_topological_dominance_sp/sp10_metric_invariance")
) -> None:
    """
    O-2: Metric distortions preserve topology → SP remains ~1.
    
    Parameters
    ----------
    n_trials : int, default=200
        Number of trials per transformation
    seed : int, default=101
        Random seed for reproducibility
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"

    rotations = [0, 30, 60, 90, 120, 180]
    scales = [(0.5, 2.0), (2.0, 0.5)]
    shears = [0.0, 0.3, 0.7, 1.0]

    records = []

    for trial in range(n_trials):
        # Rotation tests
        for th in rotations:
            coords_new = rotate_2d(base_coords, np.deg2rad(th))
            sp_val = compute_sp_total(base_coords, coords_new, layout_type=layout_type)
            records.append({
                "trial": trial,
                "transform": "rotate",
                "theta": th,
                "sp": sp_val
            })

        # Anisotropic scaling tests
        for sx, sy in scales:
            coords_new = scale_2d(base_coords, sx, sy)
            sp_val = compute_sp_total(base_coords, coords_new, layout_type=layout_type)
            records.append({
                "trial": trial,
                "transform": "scale",
                "sx": sx,
                "sy": sy,
                "sp": sp_val
            })

        # Shear tests
        for k in shears:
            coords_new = shear_2d(base_coords, k)
            sp_val = compute_sp_total(base_coords, coords_new, layout_type=layout_type)
            records.append({
                "trial": trial,
                "transform": "shear",
                "k": k,
                "sp": sp_val
            })

    out_path = out_dir / "sp10_metric_invariance_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "rotations": rotations,
                    "scales": scales,
                    "shears": shears,
                },
                "records": records
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"✅ Saved raw records to {out_path}")
    print(f"   Total records: {len(records)}")


if __name__ == "__main__":
    run_sp10_metric_invariance()
