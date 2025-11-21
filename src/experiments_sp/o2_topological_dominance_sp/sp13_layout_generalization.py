"""SP-13: Layout Generalization Test.

O-2 Topological Dominance: Verify that topology vs metric patterns
generalize across multiple layout types (grid, line, circle, random).

Expected Results:
    - Metric (shear): SP ≈ 1.0 for all layouts
    - Topology (permutation): SP << 1.0 for all layouts
    - Pattern consistent across layouts

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
from src.experiments_sp.i2_sp_instrument.sp03_layout_robustness import (
    make_line_layout,
    make_circle_layout,
    make_random_layout,
)


def run_sp13_layout_generalization(
    n_trials: int = 200,
    seed: int = 404,
    p_topology: float = 0.5,
    k_shear: float = 0.7,
    out_dir: Path = Path(
        "outputs_sp/o2_topological_dominance_sp/sp13_layout_generalization"
    ),
) -> None:
    """
    O-2: Layout generalization of topological dominance.
    
    For multiple layout types, compare SP under:
      - metric distortion (shear, topology-preserving)
      - topology disruption (coordinate permutation)
    
    We expect:
      SP_metric  ≈ 1.0
      SP_topology << 1.0
      across layouts.
    
    Parameters
    ----------
    n_trials : int, default=200
        Number of trials per layout × family
    seed : int, default=404
        Random seed for reproducibility
    p_topology : float, default=0.5
        Permutation proportion for topology family
    k_shear : float, default=0.7
        Shear coefficient for metric family
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base layouts shared with other experiments
    layouts = {
        "grid": make_grid_layout(n_side=8),
        "line": make_line_layout(),
        "circle": make_circle_layout(),
        "random": make_random_layout(64, rng=rng),
    }

    records = []

    for layout_name, base_coords in layouts.items():
        layout_type = layout_name

        for trial in range(n_trials):
            # Metric distortion: shear (topology preserved)
            coords_metric = shear_2d(base_coords, k=k_shear)
            sp_metric = compute_sp_total(
                base_coords, coords_metric, layout_type=layout_type
            )
            records.append(
                {
                    "layout": layout_name,
                    "family": "metric",
                    "level": k_shear,
                    "trial": trial,
                    "sp": sp_metric,
                }
            )

            # Topology disruption: coordinate permutation
            coords_topo = permute_coords(base_coords, rng=rng, p=p_topology)
            sp_topo = compute_sp_total(
                base_coords, coords_topo, layout_type=layout_type
            )
            records.append(
                {
                    "layout": layout_name,
                    "family": "topology",
                    "level": p_topology,
                    "trial": trial,
                    "sp": sp_topo,
                }
            )

    out_path = out_dir / "sp13_layout_generalization_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "p_topology": p_topology,
                    "k_shear": k_shear,
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
    run_sp13_layout_generalization()
