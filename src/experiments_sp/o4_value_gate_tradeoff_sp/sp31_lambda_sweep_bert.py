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

from pathlib import Path
import numpy as np
import pandas as pd

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.ssc_wrapper import compute_ssc
from src.core_sp.value_gate import apply_value_gate
from src.experiments_sp.bert_utils import load_bert_embeddings
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp31_lambda_sweep_bert(
    n_trials: int = 1000,
    seed: int = 601,
    lambda_values: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    out_dir: Path = Path("outputs_sp/sp31_lambda_sweep_bert"),
) -> None:
    """Test SSC-SP tradeoff with value gate parameter λ (BERT embeddings).
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per λ value
    seed : int, default=601
        Random seed for reproducibility
    lambda_values : tuple
        Value gate parameters to test
    out_dir : Path
        Output directory
    
    Notes
    -----
    Word shuffling is applied to create trial-to-trial variability.
    This models the cognitive variability in "which word goes where"
    when constructing a memory palace, while keeping the same word set.
    """
    rng = np.random.default_rng(seed)
    
    # Load BERT embeddings (cached after first call)
    print("Loading BERT embeddings...")
    bert_data = load_bert_embeddings(seed=seed)
    sem = bert_data["embeddings"]
    base_coords = bert_data["coords"]
    n_items = sem.shape[0]
    
    print(f"  Loaded {n_items} items with embeddings shape {sem.shape}")
    
    layout_type = "circle"  # BERT uses circle layout by default
    
    records = []
    
    for lam in lambda_values:
        print(f"  Processing λ={lam}...")
        for trial in range(n_trials):
            # Word shuffling: vary which word goes to which position
            # This creates trial-to-trial variability while keeping the same word set
            shuffle_seed = seed + 1000000 + trial
            shuffle_rng = np.random.default_rng(shuffle_seed)
            word_order = shuffle_rng.permutation(n_items)
            
            # Shuffled semantic embeddings for this trial
            sem_trial = sem[word_order]
            
            # Apply value gate to create λ-modulated coordinates
            trial_seed = seed + trial
            coords_value = apply_value_gate(
                base_coords, sem_trial, lam, seed=trial_seed, radius=1.0
            )
            
            # Compute SP and SSC
            sp_val = compute_sp_total(base_coords, coords_value, layout_type=layout_type)
            ssc_val = compute_ssc(sem_trial, coords_value)
            
            records.append({
                "lambda": lam,
                "trial": trial,
                "sp": sp_val,
                "ssc": ssc_val
            })
    
    # Compute summary statistics per λ
    summary_rows = []
    
    for lam in lambda_values:
        lam_records = [r for r in records if r["lambda"] == lam]
        
        sp_values = [r["sp"] for r in lam_records]
        ssc_values = [r["ssc"] for r in lam_records]
        
        sp_stats = compute_statistics(sp_values)
        ssc_stats = compute_statistics(ssc_values)
        
        summary_rows.append({
            "lambda": lam,
            "n": sp_stats["n"],
            "sp_mean": sp_stats["mean"],
            "sp_std": sp_stats["std"],
            "sp_ci_low": sp_stats["ci_low"],
            "sp_ci_high": sp_stats["ci_high"],
            "ssc_mean": ssc_stats["mean"],
            "ssc_std": ssc_stats["std"],
            "ssc_ci_low": ssc_stats["ci_low"],
            "ssc_ci_high": ssc_stats["ci_high"]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    save_experiment_results(
        experiment_id="sp31_lambda_sweep_bert",
        version="v2.1.0",
        parameters={
            "n_trials": n_trials,
            "seed": seed,
            "n_items": n_items,
            "lambda_values": list(lambda_values),
            "layout_type": layout_type,
            "word_shuffling": True
        },
        records=records,
        summary_df=summary_df,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_sp31_lambda_sweep_bert()
