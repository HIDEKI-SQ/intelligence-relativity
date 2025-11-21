"""SP-21: Semantic Noise vs SP & SSC.

O-3 Stress Tolerance: Verify that semantic noise may affect SSC while
SP remains ~1 (independence along semantic axis, coords fixed).

Expected Results:
    - SP ≈ 1.0 regardless of σ_sem (coords unchanged)
    - SSC behavior depends on noise (may remain ~0 or fluctuate)

Author: HIDEKI
Date: 2025-11
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.ssc_wrapper import compute_ssc
from src.core_sp.generators import generate_semantic_embeddings, add_semantic_noise
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout


def run_sp21_semantic_noise_sp_ssc(
    n_trials: int = 1000,
    seed: int = 501,
    n_items: int = 64,
    dim: int = 128,
    sigmas: tuple = (0.0, 0.1, 0.3, 0.5, 0.7),
    out_dir: Path = Path("outputs_sp/o3_stress_independence_sp_ssc/sp21_semantic_noise_sp_ssc"),
) -> None:
    """
    O-3: Semantic noise may affect SSC, SP remains ~1 (coords fixed).
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per sigma value
    seed : int, default=501
        Random seed for reproducibility
    n_items : int, default=64
        Number of items
    dim : int, default=128
        Embedding dimension
    sigmas : tuple, default=(0.0, 0.1, 0.3, 0.5, 0.7)
        Semantic noise levels
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"

    records = []

    for sigma in sigmas:
        for trial in range(n_trials):
            sem_clean = generate_semantic_embeddings(n_items, dim, rng)
            sem_noisy = add_semantic_noise(sem_clean, rng=rng, sigma=sigma)

            sp_val = compute_sp_total(base_coords, base_coords, layout_type=layout_type)
            ssc_val = compute_ssc(sem_noisy, base_coords)

            records.append(
                {
                    "sigma_sem": sigma,
                    "trial": trial,
                    "sp": sp_val,
                    "ssc": ssc_val,
                }
            )

    out_path = out_dir / "sp21_semantic_noise_sp_ssc_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "n_items": n_items,
                    "dim": dim,
                    "sigmas": list(sigmas),
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
    run_sp21_semantic_noise_sp_ssc()
