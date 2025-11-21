"""Tests for O-2 topological dominance experiments.

Validates:
    - All O-2 experiments run without errors
    - Topology vs metric separation
    - Layout generalization
"""

import pytest
import json
import shutil
from pathlib import Path
import numpy as np

from src.experiments_sp.o2_topological_dominance_sp.sp10_metric_invariance import run_sp10_metric_invariance
from src.experiments_sp.o2_topological_dominance_sp.sp11_topology_sensitivity import run_sp11_topology_sensitivity
from src.experiments_sp.o2_topological_dominance_sp.sp12_topology_vs_metric import run_sp12_topology_vs_metric
from src.experiments_sp.o2_topological_dominance_sp.sp13_layout_generalization import run_sp13_layout_generalization


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output_dir = tmp_path / "test_outputs_sp"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)


class TestSP10MetricInvariance:
    """Tests for SP-10."""
    
    def test_sp10_runs(self, temp_output_dir):
        """Test: SP-10 runs without errors."""
        out_dir = temp_output_dir / "sp10"
        run_sp10_metric_invariance(n_trials=5, seed=101, out_dir=out_dir)
        
        output_file = out_dir / "sp10_metric_invariance_raw.json"
        assert output_file.exists()
    
    def test_sp10_sp_high(self, temp_output_dir):
        """Test: Metric transforms → SP ≈ 1."""
        out_dir = temp_output_dir / "sp10"
        run_sp10_metric_invariance(n_trials=10, seed=101, out_dir=out_dir)
        
        with open(out_dir / "sp10_metric_invariance_raw.json") as f:
            data = json.load(f)
        
        sp_vals = [r["sp"] for r in data["records"]]
        mean_sp = np.mean(sp_vals)
        
        assert mean_sp > 0.85, f"Metric transforms should preserve SP, got {mean_sp}"


class TestSP11TopologySensitivity:
    """Tests for SP-11."""
    
    def test_sp11_runs(self, temp_output_dir):
        """Test: SP-11 runs without errors."""
        out_dir = temp_output_dir / "sp11"
        run_sp11_topology_sensitivity(n_trials=5, seed=202, out_dir=out_dir)
        
        output_file = out_dir / "sp11_topology_sensitivity_raw.json"
        assert output_file.exists()
    
    def test_sp11_monotonic_decrease(self, temp_output_dir):
        """Test: p↑ → SP↓."""
        out_dir = temp_output_dir / "sp11"
        run_sp11_topology_sensitivity(
            n_trials=10,
            seed=202,
            p_values=(0.0, 0.3, 0.7),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp11_topology_sensitivity_raw.json") as f:
            data = json.load(f)
        
        sp_by_p = {}
        for record in data["records"]:
            p = record["p"]
            if p not in sp_by_p:
                sp_by_p[p] = []
            sp_by_p[p].append(record["sp"])
        
        mean_sp = {p: np.mean(vals) for p, vals in sp_by_p.items()}
        
        assert mean_sp[0.0] > mean_sp[0.3] > mean_sp[0.7], \
            f"SP should decrease with p: {mean_sp}"


class TestSP12TopologyVsMetric:
    """Tests for SP-12."""
    
    def test_sp12_runs(self, temp_output_dir):
        """Test: SP-12 runs without errors."""
        out_dir = temp_output_dir / "sp12"
        run_sp12_topology_vs_metric(n_trials=5, seed=303, out_dir=out_dir)
        
        output_file = out_dir / "sp12_topology_vs_metric_raw.json"
        assert output_file.exists()
    
    def test_sp12_family_separation(self, temp_output_dir):
        """Test: Topology family << Metric family."""
        out_dir = temp_output_dir / "sp12"
        run_sp12_topology_vs_metric(
            n_trials=10,
            seed=303,
            p_values=(0.5,),
            shear_k=(0.5,),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp12_topology_vs_metric_raw.json") as f:
            data = json.load(f)
        
        topology_sp = [r["sp"] for r in data["records"] if r["family"] == "topology"]
        metric_sp = [r["sp"] for r in data["records"] if r["family"] == "metric"]
        
        mean_topo = np.mean(topology_sp)
        mean_metric = np.mean(metric_sp)
        
        assert mean_metric > mean_topo + 0.3, \
            f"Metric SP should be much higher than topology SP: {mean_metric} vs {mean_topo}"


class TestSP13LayoutGeneralization:
    """Tests for SP-13."""
    
    def test_sp13_runs(self, temp_output_dir):
        """Test: SP-13 runs without errors."""
        out_dir = temp_output_dir / "sp13"
        run_sp13_layout_generalization(n_trials=5, seed=404, out_dir=out_dir)
        
        output_file = out_dir / "sp13_layout_generalization_raw.json"
        assert output_file.exists()
    
    def test_sp13_pattern_consistent(self, temp_output_dir):
        """Test: Topology << Metric for all layouts."""
        out_dir = temp_output_dir / "sp13"
        run_sp13_layout_generalization(n_trials=10, seed=404, out_dir=out_dir)
        
        with open(out_dir / "sp13_layout_generalization_raw.json") as f:
            data = json.load(f)
        
        # Group by layout and family
        results = {}
        for record in data["records"]:
            key = (record["layout"], record["family"])
            if key not in results:
                results[key] = []
            results[key].append(record["sp"])
        
        # Check pattern for each layout
        layouts = set(r[0] for r in results.keys())
        for layout in layouts:
            metric_sp = np.mean(results[(layout, "metric")])
            topo_sp = np.mean(results[(layout, "topology")])
            
            assert metric_sp > topo_sp + 0.2, \
                f"Pattern not consistent for {layout}: metric={metric_sp}, topo={topo_sp}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
