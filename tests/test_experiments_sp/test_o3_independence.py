"""Tests for O-3 SP-SSC independence experiments.

Validates:
    - All O-3 experiments run without errors
    - SP-SSC independence patterns
    - SSC ≈ 0 at λ=0
"""

import pytest
import json
import shutil
from pathlib import Path
import numpy as np

from src.experiments_sp.o3_stress_independence_sp_ssc.sp20_coord_noise_sp_ssc import run_sp20_coord_noise_sp_ssc
from src.experiments_sp.o3_stress_independence_sp_ssc.sp21_semantic_noise_sp_ssc import run_sp21_semantic_noise_sp_ssc
from src.experiments_sp.o3_stress_independence_sp_ssc.sp22_mixed_noise_grid import run_sp22_mixed_noise_grid


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output_dir = tmp_path / "test_outputs_sp"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)


class TestSP20CoordNoise:
    """Tests for SP-20."""
    
    def test_sp20_runs(self, temp_output_dir):
        """Test: SP-20 runs without errors."""
        out_dir = temp_output_dir / "sp20"
        run_sp20_coord_noise_sp_ssc(n_trials=5, seed=500, out_dir=out_dir)
        
        output_file = out_dir / "sp20_coord_noise_sp_ssc_raw.json"
        assert output_file.exists()
    
    def test_sp20_ssc_near_zero(self, temp_output_dir):
        """Test: SSC ≈ 0 at λ=0 (O-1)."""
        out_dir = temp_output_dir / "sp20"
        run_sp20_coord_noise_sp_ssc(
            n_trials=20,
            seed=500,
            sigmas=(0.0, 0.3),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp20_coord_noise_sp_ssc_raw.json") as f:
            data = json.load(f)
        
        ssc_vals = [r["ssc"] for r in data["records"]]
        mean_abs_ssc = np.mean(np.abs(ssc_vals))
        
        assert mean_abs_ssc < 0.2, f"|SSC| should be small at λ=0, got {mean_abs_ssc}"
    
    def test_sp20_sp_decreases_with_noise(self, temp_output_dir):
        """Test: σ↑ → SP↓."""
        out_dir = temp_output_dir / "sp20"
        run_sp20_coord_noise_sp_ssc(
            n_trials=10,
            seed=500,
            sigmas=(0.0, 0.5),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp20_coord_noise_sp_ssc_raw.json") as f:
            data = json.load(f)
        
        sp_by_sigma = {}
        for record in data["records"]:
            sigma = record["sigma_coord"]
            if sigma not in sp_by_sigma:
                sp_by_sigma[sigma] = []
            sp_by_sigma[sigma].append(record["sp"])
        
        mean_sp = {s: np.mean(vals) for s, vals in sp_by_sigma.items()}
        
        assert mean_sp[0.0] > mean_sp[0.5], \
            f"SP should decrease with coord noise: {mean_sp}"


class TestSP21SemanticNoise:
    """Tests for SP-21."""
    
    def test_sp21_runs(self, temp_output_dir):
        """Test: SP-21 runs without errors."""
        out_dir = temp_output_dir / "sp21"
        run_sp21_semantic_noise_sp_ssc(n_trials=5, seed=501, out_dir=out_dir)
        
        output_file = out_dir / "sp21_semantic_noise_sp_ssc_raw.json"
        assert output_file.exists()
    
    def test_sp21_sp_stable(self, temp_output_dir):
        """Test: SP ≈ 1 (coords unchanged)."""
        out_dir = temp_output_dir / "sp21"
        run_sp21_semantic_noise_sp_ssc(
            n_trials=10,
            seed=501,
            sigmas=(0.0, 0.3, 0.5),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp21_semantic_noise_sp_ssc_raw.json") as f:
            data = json.load(f)
        
        sp_vals = [r["sp"] for r in data["records"]]
        mean_sp = np.mean(sp_vals)
        
        assert mean_sp > 0.95, f"SP should remain high (coords fixed), got {mean_sp}"


class TestSP22MixedNoiseGrid:
    """Tests for SP-22."""
    
    def test_sp22_runs(self, temp_output_dir):
        """Test: SP-22 runs without errors."""
        out_dir = temp_output_dir / "sp22"
        run_sp22_mixed_noise_grid(n_trials=5, seed=502, out_dir=out_dir)
        
        output_file = out_dir / "sp22_mixed_noise_grid_raw.json"
        assert output_file.exists()
    
    def test_sp22_independence(self, temp_output_dir):
        """Test: SP depends on σ_coord, not σ_sem."""
        out_dir = temp_output_dir / "sp22"
        run_sp22_mixed_noise_grid(
            n_trials=10,
            seed=502,
            sigmas_coord=(0.0, 0.3),
            sigmas_sem=(0.0, 0.3),
            out_dir=out_dir
        )
        
        with open(out_dir / "sp22_mixed_noise_grid_raw.json") as f:
            data = json.load(f)
        
        # Group by (σ_coord, σ_sem)
        sp_grid = {}
        for record in data["records"]:
            key = (record["sigma_coord"], record["sigma_sem"])
            if key not in sp_grid:
                sp_grid[key] = []
            sp_grid[key].append(record["sp"])
        
        # SP at (0.0, 0.0) and (0.0, 0.3) should be similar
        sp_00 = np.mean(sp_grid[(0.0, 0.0)])
        sp_03 = np.mean(sp_grid[(0.0, 0.3)])
        
        assert abs(sp_00 - sp_03) < 0.1, \
            f"SP should not depend on σ_sem: {sp_00} vs {sp_03}"
        
        # SP at (0.0, 0.0) and (0.3, 0.0) should differ
        sp_30 = np.mean(sp_grid[(0.3, 0.0)])
        
        assert abs(sp_00 - sp_30) > 0.1, \
            f"SP should depend on σ_coord: {sp_00} vs {sp_30}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
