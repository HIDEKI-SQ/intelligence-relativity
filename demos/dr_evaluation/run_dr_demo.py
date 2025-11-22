"""Main script for dimensionality reduction evaluation demo.

This demo shows how SP metrics can be used to evaluate and compare
different dimensionality reduction methods on real data (MNIST).

Usage:
    python demos/dr_evaluation/run_dr_demo.py

Output:
    demos/outputs/dr_evaluation/
        - results.csv
        - comparison_bar.png
        - embeddings_scatter.png
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from demos.dr_evaluation.load_data import load_mnist_subset
from demos.dr_evaluation.apply_dr import apply_all_methods
from demos.dr_evaluation.compute_metrics import compute_all_metrics
from demos.dr_evaluation.visualize_results import plot_comparison_bar, plot_embeddings_scatter


def main():
    """Run dimensionality reduction evaluation demo."""
    
    print("="*60)
    print("Dimensionality Reduction Evaluation Demo")
    print("="*60)
    
    # Setup output directory
    output_dir = Path("demos/outputs/dr_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    X, y = load_mnist_subset(n_per_class=50, random_state=42)
    
    # Step 2: Apply dimensionality reduction
    embeddings = apply_all_methods(X, random_state=42)
    
    # Step 3: Compute SP and SSC metrics
    results = compute_all_metrics(X, embeddings, layout_type="cluster")
    
    # Step 4: Save results
    print(f"\n💾 Saving results...")
    results_path = output_dir / "results.csv"
    results.to_csv(results_path, index=False, float_format="%.4f")
    print(f"  ✅ Saved: {results_path}")
    
    # Step 5: Visualize
    print(f"\n📊 Generating visualizations...")
    plot_comparison_bar(
        results,
        output_path=output_dir / "comparison_bar.png"
    )
    plot_embeddings_scatter(
        embeddings,
        labels=y,
        output_path=output_dir / "embeddings_scatter.png"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Demo completed successfully!")
    print(f"{'='*60}")
    print(f"\n📊 Results Summary:")
    print(results.to_string(index=False))
    print(f"\n📁 Output files:")
    print(f"  - {results_path}")
    print(f"  - {output_dir / 'comparison_bar.png'}")
    print(f"  - {output_dir / 'embeddings_scatter.png'}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
