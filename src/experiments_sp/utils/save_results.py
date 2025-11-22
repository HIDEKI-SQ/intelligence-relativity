"""Utility functions for saving experiment results in standardized format.

All SP experiments v2.0.0+ must use this module to ensure consistent output:
- raw.json: Full trial-by-trial records
- summary.csv: Aggregated statistics per condition

Author: HIDEKI
Date: 2025-11
License: MIT
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def compute_statistics(values):
    """Compute mean, std, and 95% CI from array of values.
    
    Parameters
    ----------
    values : array-like
        Numeric values to summarize
    
    Returns
    -------
    dict
        Statistics with keys: mean, std, ci_low, ci_high, n
    """
    arr = np.array(values)
    n = len(arr)
    
    if n == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": 0
        }
    
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1) if n > 1 else 0.0)
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    
    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n
    }


def save_experiment_results(
    experiment_id: str,
    version: str,
    parameters: dict,
    records: list,
    summary_df: pd.DataFrame,
    out_dir: Path
):
    """Save experiment results in standardized v2.0.0 format.
    
    Parameters
    ----------
    experiment_id : str
        Experiment identifier (e.g., 'sp00_identity_isometry')
    version : str
        Version string (e.g., 'v2.0.0')
    parameters : dict
        Experiment-wide parameters
    records : list[dict]
        Trial-by-trial records
    summary_df : pd.DataFrame
        Pre-computed summary statistics (will be saved as CSV)
    out_dir : Path
        Output directory (e.g., outputs_sp/sp00_identity_isometry/)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save raw.json
    raw_data = {
        "experiment_id": experiment_id,
        "version": version,
        "parameters": parameters,
        "records": records
    }
    
    raw_path = out_dir / "raw.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    
    raw_size_kb = raw_path.stat().st_size / 1024
    print(f"✅ Saved raw.json: {raw_path} ({raw_size_kb:.1f} KB, {len(records)} records)")
    
    # 2. Save summary.csv
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")
    
    summary_size_kb = summary_path.stat().st_size / 1024
    print(f"✅ Saved summary.csv: {summary_path} ({summary_size_kb:.1f} KB, {len(summary_df)} rows)")
