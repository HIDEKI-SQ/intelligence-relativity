"""SP-51: λ Sweep with Grid Layout (BERT Embeddings).

O-4 Value-Gated Coupling: Grid layout variant with BERT embeddings.

Notes
-----
Word shuffling is applied to create trial-to-trial variability.
This models the cognitive variability in "which word goes where"
when constructing a memory palace, while keeping the same word set.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.ssc_wrapper import compute_ssc
from src.core_sp.value_gate import apply_value_gate
from src.experiments_sp.bert_utils import load_bert_embeddings
from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import make_grid_layout
from src.experiments_sp.utils.save_results import save_experiment_results, compute_statistics


def run_sp51_lambda_tradeoff_grid_bert(
    n_trials: int = 1000,
    seed: int = 701,
    lambda_values: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    out_dir: Path = Path("outputs_sp/sp51_lambda_tradeoff_grid_bert"),
) -> None:
    """Test SSC-SP tradeoff with grid layout (BERT embeddings).
    
    Parameters
    ----------
    n_trials : int, default=1000
        Number of trials per λ value
    seed : int, default=701
        Random seed for reproducibility
    lambda_values : tuple
        Value gate parameters to test
    out_dir : Path
        Output directory
    """
    
    rng = np.random.default_rng(seed)
    
    # Load BERT embeddings
    print("Loading BERT embeddings...")
    bert_data = load_bert_embeddings(seed=seed)
    sem = bert_data["embeddings"]
    n_items = sem.shape[0]
    
    print(f"  Loaded {n_items} items with embeddings shape {sem.shape}")
    
    # Generate grid layout - use appropriate size for BERT (50 items = 7x7 + 1)
    # But to match grid structure, use closest square: 7x7=49, 8x8=64
    # Since BERT has 50 items, we'll use a partial grid
    if n_items == 50:
        # 7x8 = 56, use first 50 positions
        base_coords = make_grid_layout(n_side=7)  # 7x7 = 49
        # Add one more row position
        extra = np.array([[0.0, 7.0]], dtype=np.float64)
        base_coords = np.vstack([base_coords, extra])
    else:
        # fallback: use square grid
        n_side = int(np.ceil(np.sqrt(n_items)))
        base_coords = make_grid_layout(n_side=n_side)[:n_items]
    
    layout_type = "grid"
    
    print(f"  Grid layout: {base_coords.shape}")
    
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
            
            # Apply value gate
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
        experiment_id="sp51_lambda_tradeoff_grid_bert",
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
    run_sp51_lambda_tradeoff_grid_bert()
