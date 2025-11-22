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
from pathlib import Path
import numpy as np
import pandas as pd

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.topology_ops import permute_coords
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp42_knn_k_robustness(
    n_trials: int = 200,
    seed: int = 702,
    k_values: tuple = (3, 4, 6),
    out_dir: Path = Path("outputs_sp/sp42_knn_k_robustness"),
) -> None:
    """Test robustness of SP patterns across different k-NN k values.
    
    Parameters
    ----------
    n_trials : int, default=200
        Number of trials per k value
    seed : int, default=702
        Random seed for reproducibility
    k_values : tuple
        k-NN parameters to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"
    
    records = []
    
    for k in k_values:
        for trial in range(n_trials):
            # Apply p=0.5 topology disruption
            coords_perm = permute_coords(base_coords, rng=rng, p=0.5)
            sp_val = compute_sp_total(base_coords, coords_perm, layout_type=layout_type, k=k)
            
            records.append({
                "k": k,
                "trial": trial,
                "sp": sp_val
            })
    
    # Compute summary statistics per k
    summary_rows = []
    
    for k in k_values:
        k_records = [r for r in records if r["k"] == k]
        
        sp_values = [r["sp"] for r in k_records]
        stats = compute_statistics(sp_values)
        
        summary_rows.append({
            "k": k,
            "n": stats["n"],
            "sp_mean": stats["mean"],
            "sp_std": stats["std"],
            "sp_ci_low": stats["ci_low"],
            "sp_ci_high": stats["ci_high"]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    save_experiment_results(
        experiment_id="sp42_knn_k_robustness",
        version="v2.0.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "k_values": list(k_values)
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp42_knn_k_robustness()
