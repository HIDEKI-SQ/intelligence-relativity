"""SP-12: Topology vs Metric Direct Comparison.

O-2 Topological Dominance: Direct comparison of topology disruption
vs metric distortion effects on SP.

Expected Results:
    - Metric (shear): SP remains high (~1.0)
    - Topology (permutation): SP drops sharply
    - Clear separation between families

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


def run_sp12_topology_vs_metric(
    n_trials: int = 200,
    seed: int = 303,
    p_values: tuple = (0.0, 0.3, 0.7),
    shear_k: tuple = (0.0, 0.3, 0.7),
    out_dir: Path = Path("outputs_sp/o2_topological_dominance_sp/sp12_topology_vs_metric"),
) -> None:
    """
    O-2: Direct comparison of topology disruption vs metric distortion.
    
    Parameters
    ----------
    n_trials : int, default=200
        Number of trials per condition
    seed : int, default=303
        Random seed for reproducibility
    p_values : tuple, default=(0.0, 0.3, 0.7)
        Permutation levels for topology family
    shear_k : tuple, default=(0.0, 0.3, 0.7)
        Shear levels for metric family
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"

    records = []

    # Topology disruption (permute)
    for p in p_values:
        for trial in range(n_trials):
            coords_new = permute_coords(base_coords, rng=rng, p=p)
            sp_val = compute_sp_total(base_coords, coords_new, layout_type=layout_type)
            records.append({
                "family": "topology",
                "level": p,
                "trial": trial,
                "sp": sp_val
            })

    # Metric distortion (shear)
    for k in shear_k:
        for trial in range(n_trials):
            coords_new = shear_2d(base_coords, k)
            sp_val = compute_sp_total(base_coords, coords_new, layout_type=layout_type)
            records.append({
                "family": "metric",
                "level": k,
                "trial": trial,
                "sp": sp_val
            })

    out_path = out_dir / "sp12_topology_vs_metric_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "p_values": list(p_values),
                    "shear_k": list(shear_k),
                    "seed": seed,
                },
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"✅ Saved raw records to {out_path}")
    print(f"   Total records: {len(records)}")


if __name__ == "__main__":
    run_sp12_topology_vs_metric()
