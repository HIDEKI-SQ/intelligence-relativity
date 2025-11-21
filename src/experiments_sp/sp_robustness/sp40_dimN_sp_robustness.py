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

import json
from pathlib import Path
import numpy as np

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.metric_ops import shear_2d
from src.core_sp.topology_ops import permute_coords
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout


def run_sp40_dimN_sp_robustness(
    n_trials: int = 200,
    seed: int = 700,
    n_items_list: tuple = (16, 36, 64, 100),
    out_dir: Path = Path("outputs_sp/sp_robustness/sp40_dimN_sp_robustness"),
) -> None:
    """
    Robustness: Effect of N on SP patterns.
    
    Parameters
    ----------
    n_trials : int, default=200
        Number of trials per N × family
    seed : int, default=700
        Random seed for reproducibility
    n_items_list : tuple, default=(16, 36, 64, 100)
        Number of items to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for n_items in n_items_list:
        # Approximate grid: sqrt(n_items) × sqrt(n_items)
        n_side = int(np.sqrt(n_items))
        base_coords = make_grid_layout(n_side=n_side)
        layout_type = "grid"

        # Topology (permute) vs Metric (shear)
        for trial in range(n_trials):
            # Topology disruption
            coords_perm = permute_coords(base_coords, rng=rng, p=0.5)
            sp_topo = compute_sp_total(base_coords, coords_perm, layout_type=layout_type)

            # Metric distortion
            coords_shear = shear_2d(base_coords, k=0.7)
            sp_metric = compute_sp_total(base_coords, coords_shear, layout_type=layout_type)

            records.append(
                {
                    "n_items": n_items,
                    "trial": trial,
                    "sp_topology": sp_topo,
                    "sp_metric": sp_metric,
                }
            )

    out_path = out_dir / "sp40_dimN_sp_robustness_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "n_items_list": list(n_items_list),
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
    run_sp40_dimN_sp_robustness()
