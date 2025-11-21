"""SP-31: λ Sweep (BERT Embeddings).

O-4 Value-Gated Coupling: Validate the SSC-SP tradeoff with real-world
BERT embeddings, confirming that the pattern observed with synthetic
embeddings generalizes to pretrained semantic representations.

Expected Results:
    - λ=0: SSC ≈ 0, SP ≈ 1 (natural orthogonality preserved)
    - λ↑: SSC ↑, SP ↓ (same tradeoff pattern as synthetic)
    - Real embeddings show similar dynamics

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
from src.experiments_sp.bert_utils import load_bert_embeddings


def run_sp31_lambda_sweep_bert(
    n_trials: int = 1000,
    seed: int = 601,
    lambdas: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    out_dir: Path = Path("outputs_sp/o4_value_gate_tradeoff_sp/sp31_lambda_sweep_bert"),
) -> None:
    """
    O-4: λ sweep with BERT embeddings (SP & SSC).
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per λ value
    seed : int, default=601
        Random seed for reproducibility
    lambdas : tuple, default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        Value gate parameters to test
    out_dir : Path
        Output directory
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load BERT embeddings (cached after first call)
    print("Loading BERT embeddings...")
    bert_data = load_bert_embeddings(seed=seed)
    sem = bert_data["embeddings"]
    base_coords = bert_data["coords"]
    n_items = sem.shape[0]
    
    print(f"  Loaded {n_items} items with embeddings shape {sem.shape}")
    
    layout_type = "circle"  # BERT uses circle layout by default

    records = []

    for lam in lambdas:
        print(f"  Processing λ={lam}...")
        for trial in range(n_trials):
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

    out_path = out_dir / "sp31_lambda_sweep_bert_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "n_trials": n_trials,
                    "seed": seed,
                    "n_items": n_items,
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
    run_sp31_lambda_sweep_bert()
