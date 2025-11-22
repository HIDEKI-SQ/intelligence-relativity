"""Generate summary_all.csv - consolidated statistics from all experiments."""
import json
from pathlib import Path
import pandas as pd


def generate_summary_all(outputs_dir: Path = Path("outputs_sp")):
    """Generate summary_all.csv with actual experimental results."""
    print(f"\n{'='*60}")
    print(f"📊 Generating summary_all.csv")
    print(f"{'='*60}\n")
    
    if not outputs_dir.exists():
        print(f"❌ Directory not found: {outputs_dir}")
        return
    
    all_summaries = []
    
    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        
        # Find files with new naming pattern: {short_id}_raw.json, {short_id}_summary.csv
        raw_files = list(exp_dir.glob("*_raw.json"))
        summary_files = list(exp_dir.glob("*_summary.csv"))
        
        if not raw_files or not summary_files:
            print(f"⏭️  {exp_name}: missing files (skipped)")
            continue
        
        raw_file = raw_files[0]
        summary_file = summary_files[0]
        
        # Load experiment metadata
        with raw_file.open() as f:
            raw_data = json.load(f)
        
        experiment_id = raw_data.get("experiment_id", exp_name)
        
        # Load summary and add experiment_id column
        summary_df = pd.read_csv(summary_file)
        summary_df.insert(0, "experiment_id", experiment_id)
        
        all_summaries.append(summary_df)
        
        print(f"✅ {experiment_id}: {len(summary_df)} rows")
    
    if all_summaries:
        # Concatenate all summaries
        combined_df = pd.concat(all_summaries, ignore_index=True)
        
        output_path = outputs_dir / "summary_all.csv"
        combined_df.to_csv(output_path, index=False, float_format="%.4f")
        
        print(f"\n{'='*60}")
        print(f"✅ Saved: {output_path}")
        print(f"   Total rows: {len(combined_df)}")
        print(f"   Total experiments: {len(all_summaries)}")
        print(f"{'='*60}\n")
    else:
        print("\n⚠️  No experiments found!\n")


if __name__ == "__main__":
    generate_summary_all()
