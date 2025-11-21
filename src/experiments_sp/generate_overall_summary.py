"""Generate overall summary from all experiment outputs.

This script reads all existing raw.json files and generates a lightweight
summary with statistics only.

Author: HIDEKI
Date: 2025-11
License: MIT
"""
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def compute_stats(values):
    """Compute mean, std, 95% CI from array of values."""
    if not values:
        return None
    
    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)
    ci_lower = float(np.percentile(arr, 2.5))
    ci_upper = float(np.percentile(arr, 97.5))
    
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "n": len(values)
    }


def summarize_experiment(exp_name, records, params):
    """Summarize a single experiment based on its record structure."""
    
    if not records:
        return {"error": "No records found"}
    
    exp_summary = {
        "n_trials": params.get("n_trials", len(records)),
        "seed": params.get("seed", None),
        "n_records": len(records)
    }
    
    # Get first record to determine structure
    first = records[0]
    
    # Case 1: Simple SP only (sp00, sp01, etc.)
    if "sp" in first and "lambda" not in first and "sp_topology" not in first and "p" not in first:
        sp_values = [r["sp"] for r in records]
        exp_summary["sp"] = compute_stats(sp_values)
    
    # Case 2: SSC + SP (O-3: sp20, sp21, sp22)
    elif "ssc" in first and "sp" in first and "lambda" not in first:
        ssc_values = [r["ssc"] for r in records]
        sp_values = [r["sp"] for r in records]
        exp_summary["ssc"] = compute_stats(ssc_values)
        exp_summary["sp"] = compute_stats(sp_values)
    
    # Case 3: Lambda sweep (O-4: sp30)
    elif "lambda" in first and "ssc" in first:
        by_lambda = {}
        for r in records:
            lam = r["lambda"]
            if lam not in by_lambda:
                by_lambda[lam] = {"ssc": [], "sp": []}
            by_lambda[lam]["ssc"].append(r["ssc"])
            by_lambda[lam]["sp"].append(r["sp"])
        
        exp_summary["by_lambda"] = {}
        for lam in sorted(by_lambda.keys()):
            exp_summary["by_lambda"][str(lam)] = {
                "ssc": compute_stats(by_lambda[lam]["ssc"]),
                "sp": compute_stats(by_lambda[lam]["sp"])
            }
    
    # Case 4: Topology vs Metric (sp40)
    elif "sp_topology" in first and "sp_metric" in first:
        topo_values = [r["sp_topology"] for r in records]
        metric_values = [r["sp_metric"] for r in records]
        exp_summary["sp_topology"] = compute_stats(topo_values)
        exp_summary["sp_metric"] = compute_stats(metric_values)
    
    # Case 5: SP with p parameter (sp02, sp11, sp41)
    elif "sp" in first and "p" in first:
        by_p = {}
        for r in records:
            p = r.get("p", 0.0)
            if p not in by_p:
                by_p[p] = []
            by_p[p].append(r.get("sp", r.get("sp_adj", 0)))
        
        exp_summary["by_p"] = {}
        for p in sorted(by_p.keys()):
            exp_summary["by_p"][str(p)] = compute_stats(by_p[p])
    
    # Case 6: Multiple layouts or cases (sp03, sp13)
    elif "layout" in first or "case" in first:
        group_key = "layout" if "layout" in first else "case"
        by_group = {}
        
        for r in records:
            group = r.get(group_key, "unknown")
            if group not in by_group:
                by_group[group] = []
            by_group[group].append(r.get("sp", 0))
        
        exp_summary[f"by_{group_key}"] = {}
        for group in sorted(by_group.keys()):
            exp_summary[f"by_{group_key}"][group] = compute_stats(by_group[group])
    
    # Case 7: Transform family (sp10, sp12)
    elif "family" in first:
        by_family = {}
        for r in records:
            family = r.get("family", "unknown")
            if family not in by_family:
                by_family[family] = []
            by_family[family].append(r.get("sp", 0))
        
        exp_summary["by_family"] = {}
        for family in sorted(by_family.keys()):
            exp_summary["by_family"][family] = compute_stats(by_family[family])
    
    return exp_summary


def generate_summary(outputs_dir: Path = Path("outputs_sp")):
    """Generate overall summary from all experiment raw.json files."""
    
    print(f"\n{'='*60}")
    print(f"🔍 Generating Overall Summary")
    print(f"{'='*60}")
    print(f"📁 Scanning: {outputs_dir.absolute()}\n")
    
    if not outputs_dir.exists():
        print(f"❌ Directory not found: {outputs_dir}")
        return None
    
    summary = {
        "metadata": {
            "version": "v2.0.0",
            "generated_at": datetime.now().isoformat(),
            "source": "All *_raw.json files in outputs_sp/",
            "total_experiments": 0,
            "total_records": 0
        },
        "experiments": {}
    }
    
    # Find all experiment directories
    exp_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()])
    print(f"📂 Found {len(exp_dirs)} experiment directories\n")
    
    if not exp_dirs:
        print("⚠️  No experiment directories found!")
        return None
    
    # Process each experiment
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        raw_file = exp_dir / f"{exp_name}_raw.json"
        
        if not raw_file.exists():
            print(f"⏭️  {exp_name}: no raw.json (skipped)")
            continue
        
        try:
            print(f"✅ {exp_name}: ", end="")
            
            with raw_file.open() as f:
                data = json.load(f)
            
            params = data.get("parameters", {})
            records = data.get("records", [])
            
            file_size_kb = raw_file.stat().st_size / 1024
            print(f"{len(records)} records, {file_size_kb:.1f} KB")
            
            summary["metadata"]["total_experiments"] += 1
            summary["metadata"]["total_records"] += len(records)
            
            # Summarize this experiment
            exp_summary = summarize_experiment(exp_name, records, params)
            summary["experiments"][exp_name] = exp_summary
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Save overall summary
    output_path = outputs_dir / "overall_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    summary_size_kb = output_path.stat().st_size / 1024
    
    print(f"\n{'='*60}")
    print(f"✅ Summary saved: {output_path}")
    print(f"{'='*60}")
    print(f"📊 Total experiments: {summary['metadata']['total_experiments']}")
    print(f"📊 Total records: {summary['metadata']['total_records']}")
    print(f"📊 Summary size: {summary_size_kb:.1f} KB")
    print(f"{'='*60}\n")
    
    return summary


if __name__ == "__main__":
    generate_summary()
