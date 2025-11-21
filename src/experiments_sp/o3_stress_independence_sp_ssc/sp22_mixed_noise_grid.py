"""SP-22: Mixed Noise Grid (2D Parameter Sweep).

O-3 Stress Tolerance: 2D grid exploration of coordinate noise × semantic noise
to demonstrate independence: SP depends only on σ_coord, SSC ≈ 0 at λ=0.

Expected Results:
    - SP = f(σ_coord) only (independent of σ_sem)
    - SSC ≈ 0 for all (σ_coord, σ_sem) combinations (λ=0)

Author: HIDEKI
Date: 2025-11
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.metric_ops import add_coord_noise
from src.core_sp.ssc_wrapper import compute_ssc
from src.core_sp.generators import generate_semantic_embeddings, add_semantic_noise
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout


def run_sp22_mixed_noise_grid(
    n_trials: int = 1000,
    seed: int = 502,
    n_items: int = 64,
    dim: int = 128,
    sigmas_coord: tuple = (0.0, 0.1, 0.3, 0.5),
    sigmas_sem: tuple = (0.0, 0.1, 0.3, 0.5),
    out_dir: Path = Path("outputs_sp/o3_stress_independence_sp_ssc/sp22_mixed_noise_grid"),
) -> None:
    """
    O-3: 2D grid of (coord noise, semantic noise) with SP & SSC.
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per (σ_coord, σ_sem) pair
    seed : int, default=502
        Random seed for reproducibility
    n_items : int, default=64
        Number of items
    dim : int, default=128
        Embedding dimension
    sigmas_coord : tuple, default=(0.0, 0.1, 0.3, 0.5)
        Coordinate noise levels
    sigmas_sem : tuple, default=(0.0, 0.1, 0.3, 0.5)
        Semantic noise levels
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"

    records = []

    for sc in sigmas_coord:
        for ss in sigmas_sem:
            for trial in range(n_trials):
                sem = generate_semantic_embeddings(n_items, dim, rng)
                sem_noisy = add_semantic_noise(sem, rng=rng, sigma=ss)
                coords_noisy = add_coord_noise(base_coords, rng=rng, sigma=sc)

                sp_val = compute_sp_total(base_coords, coords_noisy, layout_type=layout_type)
                ssc_val = compute_ssc(sem_noisy, coords_noisy)

                records.append(
                    {
                        "sigma_coord": sc,
                        "sigma_sem": ss,
                        "trial": trial,
                        "sp": sp_val,
                        "ssc": ssc_val,
                    }
                )

    out_path = out_dir / "sp22_mixed_noise_grid_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "n_items": n_items,
                    "dim": dim,
                    "sigmas_coord": list(sigmas_coord),
                    "sigmas_sem": list(sigmas_sem),
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
    run_sp22_mixed_noise_grid()
