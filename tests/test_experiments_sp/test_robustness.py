"""Tests for robustness experiments.

Validates:
    - Robustness experiments run without errors
    - Patterns hold across N, dim, layout, k
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
        run_sp40_dimN_sp_robustness(n_trials=5, seed=800, out_dir=out_dir)
        
        output_file = out_dir / "sp40_dimN_sp_robustness_raw.json"
        assert output_file.exists()
    
    def test_sp40_pattern_consistent(self, temp_output_dir):
        """Test: SP values in valid range across different N."""
        out_dir = temp_output_dir / "sp40"
        run_sp40_dimN_sp_robustness(n_trials=10, seed=800, out_dir=out_dir)
        
        with open(out_dir / "sp40_dimN_sp_robustness_raw.json") as f:
            data = json.load(f)
        
        # Check that all SP values are in [0, 1] and finite
        for record in data["records"]:
            sp = record["sp"]
            assert 0.0 <= sp <= 1.0, f"SP out of range for N={record['n_items']}: {sp}"
            assert np.isfinite(sp), f"SP not finite for N={record['n_items']}"


class TestSP41LayoutTopologyRobustness:
    """Tests for SP-41."""
    
    def test_sp41_runs(self, temp_output_dir):
        """Test: SP-41 runs without errors."""
        out_dir = temp_output_dir / "sp41"
        run_sp41_layout_topology_sp_robustness(n_trials=5, seed=801, out_dir=out_dir)
        
        output_file = out_dir / "sp41_layout_topology_sp_robustness_raw.json"
        assert output_file.exists()
    
    def test_sp41_monotonic_decrease(self, temp_output_dir):
        """Test: p↑ → SP↓ across layouts."""
        out_dir = temp_output_dir / "sp41"
        run_sp41_layout_topology_sp_robustness(
            n_trials=10,
            seed=801,
            p_values=(0.0, 0.5, 1.0),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp41_layout_topology_sp_robustness_raw.json") as f:
            data = json.load(f)
        
        # Group by (layout, p)
        results = {}
        for record in data["records"]:
            key = (record["layout"], record["p"])
            if key not in results:
                results[key] = []
            results[key].append(record["sp"])
        
        # Check monotonicity for each layout
        layouts = set(k[0] for k in results.keys())
        for layout in layouts:
            sp_0 = np.mean(results[(layout, 0.0)])
            sp_05 = np.mean(results[(layout, 0.5)])
            sp_1 = np.mean(results[(layout, 1.0)])
            
            assert sp_0 > sp_05 > sp_1, \
                f"SP should decrease with p for {layout}: {sp_0}, {sp_05}, {sp_1}"


class TestSP42KNNKRobustness:
    """Tests for SP-42."""
    
    def test_sp42_runs(self, temp_output_dir):
        """Test: SP-42 runs without errors."""
        out_dir = temp_output_dir / "sp42"
        run_sp42_knn_k_robustness(n_trials=5, seed=802, out_dir=out_dir)
        
        output_file = out_dir / "sp42_knn_k_robustness_raw.json"
        assert output_file.exists()
    
    def test_sp42_sp_range(self, temp_output_dir):
        """Test: SP in [0, 1] for all k."""
        out_dir = temp_output_dir / "sp42"
        run_sp42_knn_k_robustness(n_trials=10, seed=802, out_dir=out_dir)
        
        with open(out_dir / "sp42_knn_k_robustness_raw.json") as f:
            data = json.load(f)
        
        for record in data["records"]:
            sp = record["sp"]
            assert 0.0 <= sp <= 1.0, f"SP out of range for k={record['k']}: {sp}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
