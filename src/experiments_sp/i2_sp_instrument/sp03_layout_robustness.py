"""SP-03: Layout Robustness Test.

I-2 Instrument Validation: Verify that SP instrument characteristics
(isometry invariance, destruction behavior) generalize across
multiple layout types.

Expected Results:
    - Identity: SP ≈ 1.0 for all layouts
    - Destruction: SP ≈ 0.0 for all layouts

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


def make_line_layout(n: int = 64) -> np.ndarray:
    """Generate 1D line layout."""
    xs = np.linspace(-1, 1, n)
    ys = np.zeros(n)
    return np.stack([xs, ys], axis=1)


def make_circle_layout(n: int = 64, r: float = 1.0) -> np.ndarray:
    """Generate circular layout."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def make_random_layout(n: int = 64, rng=None) -> np.ndarray:
    """Generate random layout."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(-1, 1, size=(n, 2))


def run_sp03_layout_robustness(
    n_trials: int = 100,
    seed: int = 55,
    out_dir: Path = Path("outputs_sp/i2_sp_instrument/sp03_layout_robustness"),
) -> None:
    """
    I-2: SP-03 Layout robustness.
    
    Check that isometry invariance and destruction behavior generalize
    across multiple layout types.
    
    Parameters
    ----------
    n_trials : int, default=100
        Number of trials for destruction test
    seed : int, default=55
        Random seed for reproducibility
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    layouts = {
        "grid": make_grid_layout(n_side=8),
        "line": make_line_layout(),
        "circle": make_circle_layout(),
        "random": make_random_layout(64, rng=rng),
    }

    records = []

    for layout_name, coords in layouts.items():
        # Identity test (deterministic, single trial)
        sp_identity = compute_sp_total(coords, coords, layout_type=layout_name)
        records.append(
            {
                "layout": layout_name,
                "case": "identity",
                "trial": 0,
                "sp": sp_identity,
            }
        )

        # Destruction test (n_trials)
        for trial in range(n_trials):
            coords_rand = random_relayout(coords, rng=rng, bounds=(-1, 1))
            sp_destruct = compute_sp_total(coords, coords_rand, layout_type=layout_name)
            records.append(
                {
                    "layout": layout_name,
                    "case": "destruction",
                    "trial": trial,
                    "sp": sp_destruct,
                }
            )

    out_path = out_dir / "sp03_layout_robustness_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "layouts": list(layouts.keys()),
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
    run_sp03_layout_robustness()
