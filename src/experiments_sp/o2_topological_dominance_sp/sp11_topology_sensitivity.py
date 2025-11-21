"""SP-11: Topology Sensitivity Test.

O-2 Topological Dominance: Verify that topology disruption
(coordinate permutation) causes sharp SP decrease.

Expected Results:
    - p=0.0: SP ≈ 1.0 (no disruption)
    - p↑: SP ↓ sharply (monotonic decrease)
    - p≥0.5: SP ≈ 0.0 (severe disruption)

Author: HIDEKI
Date: 2025-11
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.topology_ops import permute_coords
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout


def run_sp11_topology_sensitivity(
    n_trials: int = 200,
    seed: int = 202,
    p_values: tuple = (0.0, 0.1, 0.3, 0.5, 0.7),
    out_dir: Path = Path("outputs_sp/o2_topological_dominance_sp/sp11_topology_sensitivity"),
) -> None:
    """
    O-2: Topology disruption should sharply reduce SP.
    
    Parameters
    ----------
    n_trials : int, default=200
        Number of trials per p value
    seed : int, default=202
        Random seed for reproducibility
    p_values : tuple, default=(0.0, 0.1, 0.3, 0.5, 0.7)
        Permutation proportions to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"

    records = []

    for p in p_values:
        for trial in range(n_trials):
            coords_new = permute_coords(base_coords, rng=rng, p=p)
            sp_val = compute_sp_total(base_coords, coords_new, layout_type=layout_type)
            records.append({
                "p": p,
                "trial": trial,
                "sp": sp_val
            })

    out_path = out_dir / "sp11_topology_sensitivity_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "p_values": list(p_values),
                    "seed": seed
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
    run_sp11_topology_sensitivity()
