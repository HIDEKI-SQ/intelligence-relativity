"""Tests for I-2 instrument validation experiments.

Validates:
    - All I-2 experiments run without errors
    - Output files are created
    - Data integrity (ranges, counts)
    - Determinism
"""

import pytest
import json
import shutil
from pathlib import Path
import numpy as np

from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import run_sp00_identity_isometry
from src.experiments_sp.i2_sp_instrument.sp01_full_destruction import run_sp01_full_destruction
from src.experiments_sp.i2_sp_instrument.sp02_topology_rewire_curve import run_sp02_topology_rewire_curve
from src.experiments_sp.i2_sp_instrument.sp03_layout_robustness import run_sp03_layout_robustness


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for tests."""
    output_dir = tmp_path / "test_outputs_sp"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    # Cleanup
    if output_dir.exists():
        shutil.rmtree(output_dir)


class TestSP00IdentityIsometry:
    """Tests for SP-00."""
    
    def test_sp00_runs(self, temp_output_dir):
        """Test: SP-00 runs without errors."""
        out_dir = temp_output_dir / "sp00"
        run_sp00_identity_isometry(n_trials=5, seed=42, out_dir=out_dir)
        
        # Check output file exists
        output_file = out_dir / "sp00_identity_isometry_raw.json"
        assert output_file.exists(), "Output file not created"
    
    def test_sp00_output_structure(self, temp_output_dir):
        """Test: SP-00 output has correct structure."""
        out_dir = temp_output_dir / "sp00"
        run_sp00_identity_isometry(n_trials=5, seed=42, out_dir=out_dir)
        
        with open(out_dir / "sp00_identity_isometry_raw.json") as f:
            data = json.load(f)
        
        assert "parameters" in data
        assert "records" in data
        assert data["parameters"]["n_trials"] == 5
        assert data["parameters"]["seed"] == 42
    
    def test_sp00_sp_range(self, temp_output_dir):
        """Test: SP values in [0, 1]."""
        out_dir = temp_output_dir / "sp00"
        run_sp00_identity_isometry(n_trials=5, seed=42, out_dir=out_dir)
        
        with open(out_dir / "sp00_identity_isometry_raw.json") as f:
            data = json.load(f)
        
        for record in data["records"]:
            sp = record["sp"]
            assert 0.0 <= sp <= 1.0, f"SP out of range: {sp}"
    
    def test_sp00_determinism(self, temp_output_dir):
        """Test: Same seed → same results."""
        out_dir1 = temp_output_dir / "sp00_run1"
        out_dir2 = temp_output_dir / "sp00_run2"
        
        run_sp00_identity_isometry(n_trials=5, seed=42, out_dir=out_dir1)
        run_sp00_identity_isometry(n_trials=5, seed=42, out_dir=out_dir2)
        
        with open(out_dir1 / "sp00_identity_isometry_raw.json") as f:
            data1 = json.load(f)
        with open(out_dir2 / "sp00_identity_isometry_raw.json") as f:
            data2 = json.load(f)
        
        # Compare SP values
        sp_vals1 = [r["sp"] for r in data1["records"]]
        sp_vals2 = [r["sp"] for r in data2["records"]]
        
        assert np.allclose(sp_vals1, sp_vals2), "Results not deterministic"


class TestSP01FullDestruction:
    """Tests for SP-01."""
    
    def test_sp01_runs(self, temp_output_dir):
        """Test: SP-01 runs without errors."""
        out_dir = temp_output_dir / "sp01"
        run_sp01_full_destruction(n_trials=5, seed=123, out_dir=out_dir)
        
        output_file = out_dir / "sp01_full_destruction_raw.json"
        assert output_file.exists()
    
    def test_sp01_sp_low(self, temp_output_dir):
        """Test: Destruction → SP ≈ 0."""
        out_dir = temp_output_dir / "sp01"
        run_sp01_full_destruction(n_trials=10, seed=123, out_dir=out_dir)
        
        with open(out_dir / "sp01_full_destruction_raw.json") as f:
            data = json.load(f)
        
        sp_vals = [r["sp"] for r in data["records"]]
        mean_sp = np.mean(sp_vals)
        
        assert mean_sp < 0.3, f"Mean SP should be low for destruction, got {mean_sp}"
    
    def test_sp01_determinism(self, temp_output_dir):
        """Test: Deterministic destruction."""
        out_dir1 = temp_output_dir / "sp01_run1"
        out_dir2 = temp_output_dir / "sp01_run2"
        
        run_sp01_full_destruction(n_trials=5, seed=123, out_dir=out_dir1)
        run_sp01_full_destruction(n_trials=5, seed=123, out_dir=out_dir2)
        
        with open(out_dir1 / "sp01_full_destruction_raw.json") as f:
            data1 = json.load(f)
        with open(out_dir2 / "sp01_full_destruction_raw.json") as f:
            data2 = json.load(f)
        
        sp_vals1 = [r["sp"] for r in data1["records"]]
        sp_vals2 = [r["sp"] for r in data2["records"]]
        
        assert np.allclose(sp_vals1, sp_vals2)


class TestSP02TopologyRewire:
    """Tests for SP-02."""
    
    def test_sp02_runs(self, temp_output_dir):
        """Test: SP-02 runs without errors."""
        out_dir = temp_output_dir / "sp02"
        run_sp02_topology_rewire_curve(n_trials=5, seed=77, out_dir=out_dir)
        
        output_file = out_dir / "sp02_topology_rewire_raw.json"
        assert output_file.exists()
    
    def test_sp02_monotonic_decrease(self, temp_output_dir):
        """Test: p↑ → SP_adj↓ (monotonic)."""
        out_dir = temp_output_dir / "sp02"
        run_sp02_topology_rewire_curve(
            n_trials=10,
            seed=77,
            p_values=(0.0, 0.5, 1.0),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp02_topology_rewire_raw.json") as f:
            data = json.load(f)
        
        # Group by p
        sp_by_p = {}
        for record in data["records"]:
            p = record["p"]
            if p not in sp_by_p:
                sp_by_p[p] = []
            sp_by_p[p].append(record["sp_adj"])
        
        mean_sp = {p: np.mean(vals) for p, vals in sp_by_p.items()}
        
        assert mean_sp[0.0] > mean_sp[0.5] > mean_sp[1.0], \
            f"SP_adj should decrease monotonically: {mean_sp}"


class TestSP03LayoutRobustness:
    """Tests for SP-03."""
    
    def test_sp03_runs(self, temp_output_dir):
        """Test: SP-03 runs without errors."""
        out_dir = temp_output_dir / "sp03"
        run_sp03_layout_robustness(n_trials=5, seed=55, out_dir=out_dir)
        
        output_file = out_dir / "sp03_layout_robustness_raw.json"
        assert output_file.exists()
    
    def test_sp03_identity_high(self, temp_output_dir):
        """Test: Identity → SP ≈ 1 for all layouts."""
        out_dir = temp_output_dir / "sp03"
        run_sp03_layout_robustness(n_trials=5, seed=55, out_dir=out_dir)
        
        with open(out_dir / "sp03_layout_robustness_raw.json") as f:
            data = json.load(f)
        
        identity_records = [r for r in data["records"] if r["case"] == "identity"]
        
        for record in identity_records:
            assert record["sp"] > 0.95, \
                f"Identity SP should be high for {record['layout']}, got {record['sp']}"
    
    def test_sp03_destruction_low(self, temp_output_dir):
        """Test: Destruction → SP low for all layouts."""
        out_dir = temp_output_dir / "sp03"
        run_sp03_layout_robustness(n_trials=10, seed=55, out_dir=out_dir)
        
        with open(out_dir / "sp03_layout_robustness_raw.json") as f:
            data = json.load(f)
        
        destruction_records = [r for r in data["records"] if r["case"] == "destruction"]
        
        # Group by layout
        sp_by_layout = {}
        for record in destruction_records:
            layout = record["layout"]
            if layout not in sp_by_layout:
                sp_by_layout[layout] = []
            sp_by_layout[layout].append(record["sp"])
        
        for layout, sp_vals in sp_by_layout.items():
            mean_sp = np.mean(sp_vals)
            assert mean_sp < 0.4, f"Destruction SP should be low for {layout}, got {mean_sp}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
