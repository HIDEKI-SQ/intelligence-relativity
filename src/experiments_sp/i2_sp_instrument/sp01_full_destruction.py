"""SP-01: Full Destruction Benchmark.

I-2 Instrument Validation: Verify that complete randomization of layout
results in SP ≈ 0 (no structural preservation).

Expected Results:
    - Random relayout: SP ≈ 0 (all structural relations destroyed)

Author: HIDEKI
Date: 2025-11
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.topology_ops import random_relayout
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout


def run_sp01_full_destruction(
    n_trials: int = 100,
    seed: int = 123,
    out_dir: Path = Path("outputs_sp/i2_sp_instrument/sp01_full_destruction"),
) -> None:
    """
    I-2: SP-01 Full destruction benchmark.
    
    For each trial, completely randomize the layout and confirm SP ≈ 0.
    
    Parameters
    ----------
    n_trials : int, default=100
        Number of trials
    seed : int, default=123
        Random seed for reproducibility
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_coords = make_grid_layout(n_side=8, span=2.0)
    layout_type = "grid"

    records = []

    for trial in range(n_trials):
        coords_rand = random_relayout(base_coords, rng=rng, bounds=(-1.0, 1.0))
        sp_val = compute_sp_total(base_coords, coords_rand, layout_type=layout_type)
        records.append(
            {
                "trial": trial,
                "sp": sp_val,
            }
        )

    out_path = out_dir / "sp01_full_destruction_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "layout_type": layout_type,
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
    run_sp01_full_destruction()
