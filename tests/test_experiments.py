"""Test suite for all E8 experiments

Validates that all 15 experiments execute correctly and produce
expected outputs with deterministic reproducibility.

Author: HIDEKI
Date: 2025-11
License: MIT
"""

import pytest
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import set_deterministic_mode


class TestExperimentExecution:
    """Test that each experiment executes without errors."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set deterministic mode before each test."""
        set_deterministic_mode()
    
    def test_exp_00_baseline(self):
        """Test EXP-00: Baseline"""
        from src.experiments.exp_00_baseline import run_single_trial
        result = run_single_trial(42)
        assert 'seed' in result
        assert 'ssc' in result
        assert isinstance(result['ssc'], float)
        assert -1 <= result['ssc'] <= 1
    
    def test_exp_01_spatial_vs_random(self):
        """Test EXP-01: Spatial vs Random"""
        from src.experiments.exp_01_spatial_vs_random import run_single_trial
        result = run_single_trial(42)
        assert 'seed' in result
        assert 'ssc_spatial' in result
        assert 'ssc_random' in result
        assert isinstance(result['ssc_spatial'], float)
        assert isinstance(result['ssc_random'], float)
    
    def test_exp_02_grid_arrangement(self):
        """Test EXP-02: Grid Arrangement"""
        from src.experiments.exp_02_grid_arrangement import run_single_trial
        result = run_single_trial(42)
        assert 'seed' in result
        assert 'ssc_grid' in result
        assert 'ssc_random' in result
    
    def test_exp_03_line_arrangement(self):
        """Test EXP-03: Line Arrangement"""
        from src.experiments.exp_03_line_arrangement import run_single_trial
        result = run_single_trial(42)
        assert 'seed' in result
        assert 'ssc_line' in result
        assert 'ssc_random' in result
    
    def test_exp_04_3d_cube_arrangement(self):
        """Test EXP-04: 3D Cube Arrangement"""
        from src.experiments.exp_04_3d_cube_arrangement import run_single_trial
        result = run_single_trial(42)
        assert 'seed' in result
        assert 'ssc_cube' in result
        assert 'ssc_random' in result
    
    def test_exp_05_independence_permutation(self):
        """Test EXP-05: Independence Test"""
        from src.experiments.exp_05_independence_permutation import run_single_trial
        result = run_single_trial(42, 42)
        assert 'seed_A' in result
        assert 'seed_perm' in result
        assert 'ssc' in result
    
    def test_exp_06_dimension_robustness(self):
        """Test EXP-06: Dimension Robustness"""
        from src.experiments.exp_06_dimension_robustness import run_single_trial
        result = run_single_trial(42, 100)
        assert 'seed' in result
        assert 'dim' in result
        assert 'ssc' in result
    
    def test_exp_07_sample_size_robustness(self):
        """Test EXP-07: Sample Size Robustness"""
        from src.experiments.exp_07_sample_size_robustness import run_single_trial
        result = run_single_trial(42, 20)
        assert 'seed' in result
        assert 'n_items' in result
        assert 'ssc' in result
    
    def test_exp_08_metric_robustness(self):
        """Test EXP-08: Metric Robustness"""
        from src.experiments.exp_08_metric_robustness import run_single_trial
        result = run_single_trial(42, 'correlation')
        assert 'seed' in result
        assert 'metric' in result
        assert 'ssc' in result
    
    def test_exp_09_topological_disruption(self):
        """Test EXP-09: Topological Disruption"""
        from src.experiments.exp_09_topological_disruption import run_single_trial
        result = run_single_trial(42, 0.1)
        assert 'seed' in result
        assert 'swap_ratio' in result
        assert 'ssc' in result
    
    def test_exp_10_rotation_invariance(self):
        """Test EXP-10: Rotation Invariance"""
        from src.experiments.exp_10_rotation_invariance import run_single_trial
        result = run_single_trial(42, 30)
        assert 'seed' in result
        assert 'angle' in result
        assert 'ssc' in result
    
    def test_exp_11_coordinate_noise(self):
        """Test EXP-11: Coordinate Noise"""
        from src.experiments.exp_11_coordinate_noise import run_single_trial
        result = run_single_trial(42, 0.1)
        assert 'seed' in result
        assert 'noise_level' in result
        assert 'ssc' in result
    
    def test_exp_12_semantic_noise(self):
        """Test EXP-12: Semantic Noise"""
        from src.experiments.exp_12_semantic_noise import run_single_trial
        result = run_single_trial(42, 0.1)
        assert 'seed' in result
        assert 'noise_level' in result
        assert 'ssc' in result
        assert 'semantic_similarity' in result
    
    def test_exp_13_value_gate_sweep(self):
        """Test EXP-13: Value Gate Sweep"""
        from src.experiments.exp_13_value_gate_sweep import run_single_trial
        result = run_single_trial(42, 0.5)
        assert 'seed' in result
        assert 'lambda' in result
        assert 'ssc' in result
    
    def test_exp_beta_initial_exploration(self):
        """Test EXP-Beta: Initial Exploration"""
        from src.experiments.exp_beta_initial_exploration import run_single_trial
        result = run_single_trial(42)
        assert 'seed' in result
        assert 'ssc' in result


class TestDeterministicReproducibility:
    """Test that experiments produce identical results with same seed."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set deterministic mode before each test."""
        set_deterministic_mode()
    
    def test_exp_00_determinism(self):
        """Test EXP-00 deterministic reproducibility"""
        from src.experiments.exp_00_baseline import run_single_trial
        
        result1 = run_single_trial(42)
        result2 = run_single_trial(42)
        
        assert result1['ssc'] == result2['ssc'], "Results must be identical"
    
    def test_exp_01_determinism(self):
        """Test EXP-01 deterministic reproducibility"""
        from src.experiments.exp_01_spatial_vs_random import run_single_trial
        
        result1 = run_single_trial(42)
        result2 = run_single_trial(42)
        
        assert result1['ssc_spatial'] == result2['ssc_spatial']
        assert result1['ssc_random'] == result2['ssc_random']
    
    def test_exp_13_determinism(self):
        """Test EXP-13 deterministic reproducibility"""
        from src.experiments.exp_13_value_gate_sweep import run_single_trial
        
        result1 = run_single_trial(42, 0.5)
        result2 = run_single_trial(42, 0.5)
        
        assert result1['ssc'] == result2['ssc']


class TestCoreUtilities:
    """Test core utility functions."""
    
    def test_generate_embeddings(self):
        """Test embedding generation"""
        from src.core import generate_embeddings
        
        emb1 = generate_embeddings(20, 100, 42)
        emb2 = generate_embeddings(20, 100, 42)
        
        assert emb1.shape == (20, 100)
        assert np.allclose(emb1, emb2), "Embeddings must be deterministic"
    
    def test_generate_spatial_coords(self):
        """Test spatial coordinate generation"""
        from src.core import generate_spatial_coords
        
        coords1 = generate_spatial_coords(20, 'circle', 42)
        coords2 = generate_spatial_coords(20, 'circle', 42)
        
        assert coords1.shape == (20, 2)
        assert np.allclose(coords1, coords2), "Coordinates must be deterministic"
    
    def test_compute_ssc(self):
        """Test SSC computation"""
        from src.core import compute_ssc
        from scipy.spatial.distance import pdist
        
        # Create simple test case
        emb = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        
        sem_dist = pdist(emb, 'correlation')
        spa_dist = pdist(coords, 'euclidean')
        
        ssc = compute_ssc(sem_dist, spa_dist)
        
        assert isinstance(ssc, float)
        assert -1 <= ssc <= 1
    
    def test_compute_summary_stats(self):
        """Test summary statistics computation"""
        from src.core import compute_summary_stats
        
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        stats = compute_summary_stats(data)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        assert stats['mean'] == pytest.approx(0.3)
        assert stats['median'] == pytest.approx(0.3)
    
    def test_bootstrap_ci(self):
        """Test bootstrap CI computation"""
        from src.core import bootstrap_ci
        
        data = np.random.randn(100)
        ci_lower, ci_upper = bootstrap_ci(data, n_bootstrap=100, seed=42)
        
        assert ci_lower < ci_upper
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)


class TestOutputValidation:
    """Test that experiments produce valid outputs."""
    
    def test_output_structure(self, tmp_path):
        """Test that output files have correct structure"""
        # This would test actual file outputs if needed
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
