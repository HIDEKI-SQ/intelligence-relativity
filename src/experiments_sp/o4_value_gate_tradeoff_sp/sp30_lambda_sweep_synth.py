"""SP-30: λ Sweep (Synthetic Embeddings).

O-4 Value-Gated Coupling: Verify that increasing λ produces the expected
tradeoff: SSC ↑ (semantic-spatial coupling increases) while SP ↓
(structural preservation decreases due to semantic reordering).

Expected Results:
    - λ=0: SSC ≈ 0, SP ≈ 1 (natural orthogonality, structure preserved)
    - λ↑: SSC ↑, SP ↓ (value-gated coupling, tradeoff emerges)
    - Linear regime followed by saturation

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
from src.core_sp.value_gate import apply_value_gate
from src.core_sp.generators import generate_semantic_embeddings
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout


def run_sp30_lambda_sweep_synth(
    n_trials: int = 1000,
    seed: int = 600,
    n_items: int = 64,
    dim: int = 128,
    lambdas: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    out_dir: Path = Path("outputs_sp/o4_value_gate_tradeoff_sp/sp30_lambda_sweep_synth"),
) -> None:
    """
    O-4: λ sweep (synthetic) with SP & SSC.
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per λ value
    seed : int, default=600
        Random seed for reproducibility
    n_items : int, default=64
        Number of items
    dim : int, default=128
        Embedding dimension
    lambdas : tuple, default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        Value gate parameters to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_coords = make_grid_layout(n_side=8)
    layout_type = "grid"

    records = []

    for lam in lambdas:
        for trial in range(n_trials):
            sem = generate_semantic_embeddings(n_items, dim, rng)
            
            # Apply value gate to create λ-modulated coordinates
            trial_seed = seed + trial
            coords_value = apply_value_gate(
                base_coords, sem, lam, seed=trial_seed, radius=1.0
            )

            sp_val = compute_sp_total(base_coords, coords_value, layout_type=layout_type)
            ssc_val = compute_ssc(sem, coords_value)

            records.append(
                {
                    "lambda": lam,
                    "trial": trial,
                    "sp": sp_val,
                    "ssc": ssc_val,
                }
            )

    out_path = out_dir / "sp30_lambda_sweep_synth_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "n_items": n_items,
                    "dim": dim,
                    "lambdas": list(lambdas),
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
    run_sp30_lambda_sweep_synth()
