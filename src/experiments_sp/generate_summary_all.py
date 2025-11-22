"""Generate summary_all.csv - a catalog of all experiments.

This script scans outputs_sp/ and creates a master index showing:
- Which experiments have been run
- Where to find their summary.csv and raw.json
- Key metadata about each experiment

Author: HIDEKI
Date: 2025-11
License: MIT
"""
import json
from pathlib import Path
import pandas as pd


def generate_summary_all(outputs_dir: Path = Path("outputs_sp")):
    """Generate summary_all.csv cataloging all experiments.
    
    Parameters
    ----------
    outputs_dir : Path
        Root directory containing experiment subdirectories
    """
    print(f"\n{'='*60}")
    print(f"📊 Generating summary_all.csv")
    print(f"{'='*60}\n")
    
    if not outputs_dir.exists():
        print(f"❌ Directory not found: {outputs_dir}")
        return
    
    catalog = []
    
    # Scan all experiment directories
    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        
        # Check for required files
        raw_file = exp_dir / "raw.json"
        summary_file = exp_dir / "summary.csv"
        
        if not raw_file.exists():
            print(f"⏭️  {exp_name}: no raw.json (skipped)")
            continue
        
        if not summary_file.exists():
            print(f"⏭️  {exp_name}: no summary.csv (skipped)")
            continue
        
        # Load metadata from raw.json
        with raw_file.open() as f:
            data = json.load(f)
        
        experiment_id = data.get("experiment_id", exp_name)
        version = data.get("version", "unknown")
        parameters = data.get("parameters", {})
        n_records = len(data.get("records", []))
        
        # Load summary to count conditions
        summary_df = pd.read_csv(summary_file)
        n_conditions = len(summary_df)
        
        # Detect if experiment has SSC
        has_ssc = "ssc_mean" in summary_df.columns
        
        # Identify primary factors (condition columns)
        exclude_cols = ["n", "mean", "std", "ci_low", "ci_high"]
        primary_factors = [
            col for col in summary_df.columns
            if not any(x in col for x in exclude_cols)
        ]
        
        catalog.append({
            "experiment_id": experiment_id,
            "version": version,
            "summary_file": f"{exp_name}/summary.csv",
            "raw_file": f"{exp_name}/raw.json",
            "n_records": n_records,
            "n_conditions": n_conditions,
            "has_ssc": has_ssc,
            "primary_factors": ",".join(primary_factors)
        })
        
        print(f"✅ {experiment_id}: {n_conditions} conditions, {n_records} records")
    
    # Save catalog
    if catalog:
        catalog_df = pd.DataFrame(catalog)
        catalog_path = outputs_dir / "summary_all.csv"
        catalog_df.to_csv(catalog_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"✅ Saved: {catalog_path}")
        print(f"   Total experiments: {len(catalog)}")
        print(f"{'='*60}\n")
    else:
        print("\n⚠️  No experiments found!\n")


if __name__ == "__main__":
    generate_summary_all()
