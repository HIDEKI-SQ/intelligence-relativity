"""SP-42: k-NN Parameter Robustness Test.

Robustness Validation: Verify that SP_adj patterns remain consistent
across different k values in k-NN adjacency graph construction.

Expected Results:
    - p=0.5 topology disruption reduces SP for all k values
    - Pattern consistent across k (though absolute values may shift)

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


def run_sp42_knn_k_robustness(
    n_trials: int = 200,
    seed: int = 702,
    k_values: tuple = (3, 4, 6),
    out_dir: Path = Path("outputs_sp/sp_robustness/sp42_knn_k_robustness"),
) -> None:
    """
    Robustness: Effect of k in k-NN graph on SP patterns.
    
    Parameters
    ----------
    n_trials : int, default=200
        Number of trials per k value
    seed : int, default=702
        Random seed for reproducibility
    k_values : tuple, default=(3, 4, 6)
        k-NN parameters to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"

    records = []

    for k in k_values:
        for trial in range(n_trials):
            coords_perm = permute_coords(base_coords, rng=rng, p=0.5)
            sp_val = compute_sp_total(base_coords, coords_perm, layout_type=layout_type, k=k)
            records.append(
                {
                    "k": k,
                    "trial": trial,
                    "sp": sp_val,
                }
            )

    out_path = out_dir / "sp42_knn_k_robustness_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "k_values": list(k_values),
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
    run_sp42_knn_k_robustness()
