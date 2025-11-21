"""SP-02: Controlled Topology Disruption (Edge Rewiring Curve).

I-2 Instrument Validation: Verify that SP_adj (adjacency preservation)
decreases monotonically as edge rewiring proportion increases.

Expected Results:
    - p=0.0: SP_adj ≈ 1.0 (no rewiring)
    - p↑: SP_adj ↓ (monotonic decrease)
    - p=1.0: SP_adj ≈ 0.0 (complete rewiring)

Note: This experiment focuses specifically on SP_adj (adjacency component)
rather than composite SP, to isolate the effect of edge rewiring.

Author: HIDEKI
Date: 2025-11
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.core_sp.sp_metrics import compute_knn_graph, jaccard_similarity_bool
from src.core_sp.topology_ops import edge_rewire
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout


def run_sp02_topology_rewire_curve(
    n_trials: int = 200,
    seed: int = 77,
    p_values: tuple = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
    k: int = 4,
    out_dir: Path = Path("outputs_sp/i2_sp_instrument/sp02_topology_rewire_curve"),
) -> None:
    """
    I-2: SP-02 Controlled topology disruption.
    
    Measure SP_adj as a function of edge rewiring proportion.
    
    Parameters
    ----------
    n_trials : int, default=200
        Number of trials per p value
    seed : int, default=77
        Random seed for reproducibility
    p_values : tuple, default=(0.0, 0.1, 0.3, 0.5, 0.7, 1.0)
        Edge rewiring proportions to test
    k : int, default=4
        Number of nearest neighbors for adjacency graph
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_coords = make_grid_layout(n_side=8, span=2.0)

    # Compute base adjacency graph
    adj_base = compute_knn_graph(base_coords, k=k)

    records = []

    for p in p_values:
        for trial in range(n_trials):
            # Rewire edges
            adj_new = edge_rewire(adj_base, p=p, rng=rng)
            
            # Compute SP_adj directly (Jaccard similarity)
            sp_adj = jaccard_similarity_bool(adj_base, adj_new)
            
            records.append(
                {
                    "p": p,
                    "trial": trial,
                    "sp_adj": sp_adj,
                }
            )

    out_path = out_dir / "sp02_topology_rewire_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "p_values": list(p_values),
                    "k": k,
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
    run_sp02_topology_rewire_curve()
