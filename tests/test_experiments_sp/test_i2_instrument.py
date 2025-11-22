"""Tests for I-2 instrument validation experiments (v2.0.0 format).

Validates:
    - Experiments run without errors
    - Generate raw.json and summary.csv
    - Output structure matches v2.0.0 spec
"""
import pytest
import json
from pathlib import Path
import shutil
import pandas as pd

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
    if output_dir.exists():
        shutil.rmtree(output_dir)


def validate_v2_output(exp_dir: Path, experiment_id: str):
    """Validate that experiment output matches v2.0.0 spec."""
    # Check files exist
    raw_file = exp_dir / "raw.json"
    summary_file = exp_dir / "summary.csv"
    
    assert raw_file.exists(), f"raw.json not found in {exp_dir}"
    assert summary_file.exists(), f"summary.csv not found in {exp_dir}"
    
    # Validate raw.json structure
    with raw_file.open() as f:
        raw_data = json.load(f)
    
    assert "experiment_id" in raw_data
    assert raw_data["experiment_id"] == experiment_id
    assert "version" in raw_data
    assert "parameters" in raw_data
    assert "records" in raw_data
    assert isinstance(raw_data["records"], list)
    assert len(raw_data["records"]) > 0
    
    # Validate summary.csv structure
    summary_df = pd.read_csv(summary_file)
    assert len(summary_df) > 0
    assert "n" in summary_df.columns
    
    return raw_data, summary_df


class TestSP00IdentityIsometry:
    """Tests for SP-00."""
    
    def test_sp00_runs(self, temp_output_dir):
        """Test: SP-00 runs without errors."""
        out_dir = temp_output_dir / "sp00_identity_isometry"
        run_sp00_identity_isometry(n_trials=1, seed=42, out_dir=out_dir)
        
        raw_data, summary_df = validate_v2_output(out_dir, "sp00_identity_isometry")
    
    def test_sp00_output_structure(self, temp_output_dir):
        """Test: SP-00 has correct columns."""
        out_dir = temp_output_dir / "sp00_identity_isometry"
        run_sp00_identity_isometry(n_trials=1, seed=42, out_dir=out_dir)
        
        summary_df = pd.read_csv(out_dir / "summary.csv")
        
        required_cols = ["layout", "transform", "n", "sp_mean", "sp_std", 
                        "sp_ci_low", "sp_ci_high"]
        for col in required_cols:
            assert col in summary_df.columns, f"Missing column: {col}"


class TestSP01FullDestruction:
    """Tests for SP-01."""
    
    def test_sp01_runs(self, temp_output_dir):
        """Test: SP-01 runs without errors."""
        out_dir = temp_output_dir / "sp01_full_destruction"
        run_sp01_full_destruction(n_trials=10, seed=123, out_dir=out_dir)
        
        raw_data, summary_df = validate_v2_output(out_dir, "sp01_full_destruction")
        
        # Check we have 10 records
        assert len(raw_data["records"]) == 10
    
    def test_sp01_sp_reduced(self, temp_output_dir):
        """Test: Destruction reduces SP."""
        out_dir = temp_output_dir / "sp01_full_destruction"
        run_sp01_full_destruction(n_trials=100, seed=123, out_dir=out_dir)
        
        summary_df = pd.read_csv(out_dir / "summary.csv")
        sp_mean = summary_df.iloc[0]["sp_mean"]
        
        assert sp_mean < 0.7, f"SP should be reduced, got {sp_mean}"


class TestSP02TopologyRewire:
    """Tests for SP-02."""
    
    def test_sp02_runs(self, temp_output_dir):
        """Test: SP-02 runs without errors."""
        out_dir = temp_output_dir / "sp02_topology_rewire_curve"
        run_sp02_topology_rewire_curve(n_trials=10, seed=77, out_dir=out_dir)
        
        raw_data, summary_df = validate_v2_output(out_dir, "sp02_topology_rewire_curve")
    
    def test_sp02_monotonic_decrease(self, temp_output_dir):
        """Test: p↑ → SP_adj↓."""
        out_dir = temp_output_dir / "sp02_topology_rewire_curve"
        run_sp02_topology_rewire_curve(
            n_trials=50,
            seed=77,
            p_values=(0.0, 0.5, 1.0),
            out_dir=out_dir
        )
        
        summary_df = pd.read_csv(out_dir / "summary.csv")
        
        sp_at_p0 = summary_df[summary_df["p"] == 0.0].iloc[0]["sp_adj_mean"]
        sp_at_p05 = summary_df[summary_df["p"] == 0.5].iloc[0]["sp_adj_mean"]
        sp_at_p1 = summary_df[summary_df["p"] == 1.0].iloc[0]["sp_adj_mean"]
        
        assert sp_at_p0 > sp_at_p05 > sp_at_p1, \
            f"SP_adj should decrease: {sp_at_p0}, {sp_at_p05}, {sp_at_p1}"


class TestSP03LayoutRobustness:
    """Tests for SP-03."""
    
    def test_sp03_runs(self, temp_output_dir):
        """Test: SP-03 runs without errors."""
        out_dir = temp_output_dir / "sp03_layout_robustness"
        run_sp03_layout_robustness(n_trials=10, seed=55, out_dir=out_dir)
        
        raw_data, summary_df = validate_v2_output(out_dir, "sp03_layout_robustness")
    
    def test_sp03_all_layouts(self, temp_output_dir):
        """Test: All layouts present."""
        out_dir = temp_output_dir / "sp03_layout_robustness"
        run_sp03_layout_robustness(n_trials=10, seed=55, out_dir=out_dir)
        
        summary_df = pd.read_csv(out_dir / "summary.csv")
        
        layouts = summary_df["layout"].unique()
        assert set(layouts) == {"grid", "line", "circle", "random"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
