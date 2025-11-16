"""Tests for experiment execution.

Validates that experiments can execute successfully and produce
expected outputs.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExperimentStructure:
    """Test experiment file structure and imports."""
    
    def test_all_experiments_importable(self):
        """Verify all experiment modules can be imported."""
        experiments = [
            "exp_00_baseline",
            "exp_01_spatial_vs_random",
            "exp_02_grid_arrangement",
            "exp_03_line_arrangement",
            "exp_04_3d_cube_arrangement",
            "exp_05_independence_permutation",
            "exp_06_dimension_robustness",
            "exp_07_sample_size_robustness",
            "exp_08_metric_robustness",
            "exp_09_topological_disruption",
            "exp_10_rotation_invariance",
            "exp_11_coordinate_noise",
            "exp_12_semantic_noise",
            "exp_13_value_gate_sweep",
            "sup_exp_14_bert",
            "sup_exp_15_multilingual",
            "exp_beta_initial_exploration",
        ]
        
        for exp_name in experiments:
            try:
                module = __import__(f"src.experiments.{exp_name}", fromlist=[''])
                assert hasattr(module, 'main'), f"{exp_name} missing main()"
            except ImportError as e:
                pytest.fail(f"Failed to import {exp_name}: {e}")
    
    def test_experiments_have_required_functions(self):
        """Verify experiments have required structure."""
        from src.experiments import exp_00_baseline
        
        # Should have main function
        assert hasattr(exp_00_baseline, 'main')
        assert callable(exp_00_baseline.main)


class TestBaselineExperiment:
    """Test EXP-00 baseline experiment."""
    
    def test_exp00_parameters(self):
        """Verify EXP-00 uses correct parameters."""
        from src.experiments import exp_00_baseline
        
        assert exp_00_baseline.N_ITEMS == 20
        assert exp_00_baseline.DIM == 100
        assert exp_00_baseline.N_TRIALS == 1000
        assert exp_00_baseline.BASE_SEED == 42
    
    def test_exp00_output_structure(self, tmp_path, monkeypatch):
        """Verify EXP-00 produces correct output structure."""
        from src.experiments import exp_00_baseline
        
        # Redirect output to temp directory
        test_output = tmp_path / "exp00_baseline"
        monkeypatch.setattr(exp_00_baseline, 'OUTPUT_DIR', test_output)
        
        # Run with small sample
        original_trials = exp_00_baseline.N_TRIALS
        monkeypatch.setattr(exp_00_baseline, 'N_TRIALS', 10)
        
        exp_00_baseline.run_exp00()
        
        # Verify outputs exist
        assert (test_output / "exp00_baseline_summary.json").exists()
        assert (test_output / "exp00_baseline_results.csv").exists()
        assert (test_output / "exp00_baseline_histogram.png").exists()
        assert (test_output / "sha256_manifest.json").exists()


class TestSupplementaryExperiments:
    """Test supplementary experiments can be imported."""
    
    def test_sup14_bert_import(self):
        """Verify SUP-14 BERT experiment can be imported."""
        try:
            from src.experiments import sup_exp_14_bert
            assert hasattr(sup_exp_14_bert, 'main')
        except ImportError:
            pytest.skip("Transformers not installed")
    
    def test_sup15_multilingual_import(self):
        """Verify SUP-15 multilingual experiment can be imported."""
        try:
            from src.experiments import sup_exp_15_multilingual
            assert hasattr(sup_exp_15_multilingual, 'main')
        except ImportError:
            pytest.skip("Transformers not installed")


class TestExperimentParameterConsistency:
    """Test parameter consistency across experiments."""
    
    def test_base_seed_consistency(self):
        """Verify BASE_SEED is consistent across experiments."""
        from src.experiments import (
            exp_00_baseline,
            exp_01_spatial_vs_random,
            exp_02_grid_arrangement
        )
        
        assert exp_00_baseline.BASE_SEED == 42
        assert exp_01_spatial_vs_random.BASE_SEED == 42
        assert exp_02_grid_arrangement.BASE_SEED == 42
    
    def test_n_items_consistency(self):
        """Verify N_ITEMS is consistent for core experiments."""
        from src.experiments import (
            exp_00_baseline,
            exp_01_spatial_vs_random,
            exp_02_grid_arrangement
        )
        
        assert exp_00_baseline.N_ITEMS == 20
        assert exp_01_spatial_vs_random.N_ITEMS == 20
        assert exp_02_grid_arrangement.N_ITEMS == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
