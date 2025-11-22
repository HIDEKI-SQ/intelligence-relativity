"""Generate series-specific summary files - consolidated statistics by experiment group."""
import json
from pathlib import Path
import pandas as pd

# Define experiment series with their experiment ID prefixes
SERIES = {
    "I2": ["sp00", "sp01", "sp02", "sp03"],
    "O2": ["sp10", "sp11", "sp12", "sp13"],
    "O3": ["sp20", "sp21", "sp22"],
    "O4": ["sp30", "sp31"],
    "robust": ["sp40", "sp41", "sp42"]
}

def generate_summary_all(outputs_dir: Path = Path("outputs_sp")):
    """Generate series-specific summary_all files.
    
    Creates separate summary files for each experiment series to avoid
    sparse tables with many empty columns. Each series has similar
    experimental conditions and column structures.
    
    Output files:
        - summary_all_I2.csv: Measurement system experiments (sp00-sp03)
        - summary_all_O2.csv: Topological dominance experiments (sp10-sp13)
        - summary_all_O3.csv: Stress tolerance experiments (sp20-sp22)
        - summary_all_O4.csv: Value-gating experiments (sp30-sp31)
        - summary_all_robust.csv: Robustness experiments (sp40-sp42)
    """
    print(f"\n{'='*60}")
    print(f"📊 Generating series-specific summary files")
    print(f"{'='*60}\n")
    
    if not outputs_dir.exists():
        print(f"❌ Directory not found: {outputs_dir}")
        return
    
    # Process each series
    for series_name, exp_prefixes in SERIES.items():
        series_summaries = []
        
        for exp_dir in sorted(outputs_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            exp_name = exp_dir.name
            
            # Check if this experiment belongs to current series
            if not any(exp_name.startswith(prefix) for prefix in exp_prefixes):
                continue
            
            # Find summary and raw files
            summary_files = list(exp_dir.glob("*_summary.csv"))
            raw_files = list(exp_dir.glob("*_raw.json"))
            
            if not summary_files or not raw_files:
                print(f"⏭️  {exp_name}: missing files (skipped)")
                continue
            
            # Load experiment ID from raw data
            with raw_files[0].open() as f:
                raw_data = json.load(f)
            experiment_id = raw_data.get("experiment_id", exp_name)
            
            # Load summary and add experiment_id column
            summary_df = pd.read_csv(summary_files[0])
            summary_df.insert(0, "experiment_id", experiment_id)
            
            series_summaries.append(summary_df)
            print(f"  ✅ {experiment_id}: {len(summary_df)} rows")
        
        if series_summaries:
            # Concatenate all summaries in this series
            combined_df = pd.concat(series_summaries, ignore_index=True)
            
            # Save series-specific summary
            output_path = outputs_dir / f"summary_all_{series_name}.csv"
            combined_df.to_csv(output_path, index=False, float_format="%.4f")
            
            print(f"\n📄 Saved: {output_path.name}")
            print(f"   Rows: {len(combined_df)}, Experiments: {len(series_summaries)}")
            print(f"{'='*60}\n")
        else:
            print(f"\n⚠️  No experiments found for series: {series_name}\n")

if __name__ == "__main__":
    generate_summary_all()
