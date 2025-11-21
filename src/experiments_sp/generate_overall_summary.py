"""Generate overall summary from all experiment outputs.

This script collects all raw.json files and generates a unified summary
with statistics for all experiments.

Author: HIDEKI
Date: 2025-11
License: MIT
"""
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def compute_stats(values):
    """Compute mean, std, 95% CI."""
    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    ci_lower = float(np.percentile(arr, 2.5))
    ci_upper = float(np.percentile(arr, 97.5))
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)]
    }


def generate_summary(outputs_dir: Path = Path("outputs_sp")):
    """Generate overall summary from all experiments."""
    
    summary = {
        "metadata": {
            "version": "v2.0.0",
            "generated_at": datetime.now().isoformat(),
            "total_experiments": 0,
            "total_records": 0
        },
        "experiments": {}
    }
    
    # Collect all raw.json files
    for exp_dir in sorted(outputs_dir.glob("sp*")):
        if not exp_dir.is_dir():
            continue
            
        raw_file = exp_dir / f"{exp_dir.name}_raw.json"
        if not raw_file.exists():
            continue
        
        with raw_file.open() as f:
            data = json.load(f)
        
        exp_name = exp_dir.name
        params = data.get("parameters", {})
        records = data.get("records", [])
        
        summary["metadata"]["total_experiments"] += 1
        summary["metadata"]["total_records"] += len(records)
        
        # Basic info
        exp_summary = {
            "n_trials": params.get("n_trials", len(records)),
            "seed": params.get("seed", None)
        }
        
        # Extract statistics based on experiment type
        if "sp" in records[0]:
            # Simple SP experiments
            sp_values = [r["sp"] for r in records]
            exp_summary["sp"] = compute_stats(sp_values)
        
        elif "ssc" in records[0] and "sp" in records[0]:
            # SSC-SP experiments (O3, O4)
            ssc_values = [r["ssc"] for r in records]
            sp_values = [r["sp"] for r in records]
            exp_summary["ssc"] = compute_stats(ssc_values)
            exp_summary["sp"] = compute_stats(sp_values)
        
        elif "lambda" in records[0]:
            # Lambda sweep (O4)
            by_lambda = {}
            for r in records:
                lam = r["lambda"]
                if lam not in by_lambda:
                    by_lambda[lam] = {"ssc": [], "sp": []}
                by_lambda[lam]["ssc"].append(r["ssc"])
                by_lambda[lam]["sp"].append(r["sp"])
            
            exp_summary["by_lambda"] = {
                str(lam): {
                    "ssc": compute_stats(vals["ssc"]),
                    "sp": compute_stats(vals["sp"])
                }
                for lam, vals in sorted(by_lambda.items())
            }
        
        elif "sp_topology" in records[0]:
            # Topology vs Metric (SP40)
            topo_values = [r["sp_topology"] for r in records]
            metric_values = [r["sp_metric"] for r in records]
            exp_summary["sp_topology"] = compute_stats(topo_values)
            exp_summary["sp_metric"] = compute_stats(metric_values)
        
        summary["experiments"][exp_name] = exp_summary
    
    # Save summary
    output_path = outputs_dir / "overall_summary.json"
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Overall summary saved to {output_path}")
    print(f"   Total experiments: {summary['metadata']['total_experiments']}")
    print(f"   Total records: {summary['metadata']['total_records']}")
    
    return summary


if __name__ == "__main__":
    generate_summary()
