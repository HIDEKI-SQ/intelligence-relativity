"""Tests for robustness validation experiments.

Validates:
    - All robustness experiments run without errors
    - Pattern consistency across parameters
"""

import pytest
import json
import shutil
from pathlib import Path
import numpy as np

from src.experiments_sp.sp_robustness.sp40_dimN_sp_robustness import run_sp40_dimN_sp_robustness
from src.experiments_sp.sp_robustness.sp41_layout_topology_sp_robustness import run_sp41_layout_topology_sp_robustness
from src.experiments_sp.sp_robustness.sp42_knn_k_robustness import run_sp42_knn_k_robustness


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output_dir = tmp_path / "test_outputs_sp"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)


class TestSP40DimNRobustness:
    """Tests for SP-40."""
    
    def test_sp40_runs(self, temp_output_dir):
        """Test: SP-40 runs without errors."""
        out_dir = temp_output_dir / "sp40"
        run_sp40_dimN_sp_robustness(n_trials=5, seed=700, out_dir=out_dir)
        
        output_file = out_dir / "sp40_dimN_sp_robustness_raw.json"
        assert output_file.exists()
    
    def test_sp40_pattern_consistent(self, temp_output_dir):
        """Test: Topology << Metric for all N."""
        out_dir = temp_output_dir / "sp40"
        run_sp40_dimN_sp_robustness(
            n_trials=10,
            seed=700,
            n_items_list=(16, 64),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp40_dimN_sp_robustness_raw.json") as f:
            data = json.load(f)
        
        # Group by N
        results = {}
        for record in data["records"]:
            n = record["n_items"]
            if n not in results:
                results[n] = {"topo": [], "metric": []}
            results[n]["topo"].append(record["sp_topology"])
            results[n]["metric"].append(record["sp_metric"])
        
        for n, sp_dict in results.items():
            mean_topo = np.mean(sp_dict["topo"])
            mean_metric = np.mean(sp_dict["metric"])
            
            assert mean_metric > mean_topo + 0.2, \
                f"Pattern not consistent for N={n}: metric={mean_metric}, topo={mean_topo}"


class TestSP41LayoutTopologyRobustness:
    """Tests for SP-41."""
    
    def test_sp41_runs(self, temp_output_dir):
        """Test: SP-41 runs without errors."""
        out_dir = temp_output_dir / "sp41"
        run_sp41_layout_topology_sp_robustness(n_trials=5, seed=701, out_dir=out_dir)
        
        output_file = out_dir / "sp41_layout_topology_sp_robustness_raw.json"
        assert output_file.exists()
    
    def test_sp41_monotonic_decrease(self, temp_output_dir):
        """Test: p↑ → SP↓ for all layouts."""
        out_dir = temp_output_dir / "sp41"
        run_sp41_layout_topology_sp_robustness(
            n_trials=10,
            seed=701,
            p_values=(0.0, 0.3, 0.7),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp41_layout_topology_sp_robustness_raw.json") as f:
            data = json.load(f)
        
        # Group by layout and p
        results = {}
        for record in data["records"]:
            layout = record["layout"]
            p = record["p"]
            if layout not in results:
                results[layout] = {}
            if p not in results[layout]:
                results[layout][p] = []
            results[layout][p].append(record["sp"])
        
        for layout, sp_by_p in results.items():
            mean_sp = {p: np.mean(vals) for p, vals in sp_by_p.items()}
            
            assert mean_sp[0.0] > mean_sp[0.3] > mean_sp[0.7], \
                f"SP should decrease monotonically for {layout}: {mean_sp}"


class TestSP42KNNKRobustness:
    """Tests for SP-42."""
    
    def test_sp42_runs(self, temp_output_dir):
        """Test: SP-42 runs without errors."""
        out_dir = temp_output_dir / "sp42"
        run_sp42_knn_k_robustness(n_trials=5, seed=702, out_dir=out_dir)
        
        output_file = out_dir / "sp42_knn_k_robustness_raw.json"
        assert output_file.exists()
    
    def test_sp42_sp_range(self, temp_output_dir):
        """Test: SP in [0, 1] for all k."""
        out_dir = temp_output_dir / "sp42"
        run_sp42_knn_k_robustness(
            n_trials=10,
            seed=702,
            k_values=(3, 4, 6),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp42_knn_k_robustness_raw.json") as f:
            data = json.load(f)
        
        for record in data["records"]:
            sp = record["sp"]
            assert 0.0 <= sp <= 1.0, f"SP out of range for k={record['k']}: {sp}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
