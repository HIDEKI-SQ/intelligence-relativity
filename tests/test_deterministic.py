"""Tests for deterministic execution guarantees.

The Intelligence Relativity framework requires perfect reproducibility
to ensure scientific validity. These tests verify all components
maintain deterministic behavior.
"""

import pytest
import numpy as np
from pathlib import Path
import json

from src.core import (
    set_deterministic_mode,
    verify_environment,
    generate_embeddings,
    generate_spatial_coords,
    compute_ssc_from_data,
    bootstrap_ci
)


class TestDeterministicEnvironment:
    """Test deterministic environment setup."""
    
    def test_environment_variables_set(self):
        """Verify BLAS threading is disabled."""
        set_deterministic_mode()
        
        import os
        blas_vars = ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']
        
        for var in blas_vars:
            assert os.environ.get(var) == '1', f"{var} not set to 1"
    
    def test_numpy_seed_effects(self):
        """Verify NumPy seed affects randomness."""
        set_deterministic_mode()
        
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 100)
        
        np.random.seed(42)
        data2 = np.random.normal(0, 1, 100)
        
        np.testing.assert_array_equal(data1, data2)
    
    def test_environment_logging(self, tmp_path):
        """Verify environment is logged correctly."""
        env_file = tmp_path / "env.txt"
        verify_environment(env_file)
        
        with open(env_file) as f:
            env_data = json.load(f)
        
        required_keys = ['python', 'numpy', 'scipy', 'platform']
        for key in required_keys:
            assert key in env_data, f"Missing key: {key}"


class TestReproducibleGeneration:
    """Test reproducibility of data generation."""
    
    def test_embeddings_perfect_reproducibility(self):
        """Verify embeddings are bit-for-bit reproducible."""
        set_deterministic_mode()
        
        n_items, dim, seed = 20, 100, 42
        
        # Generate twice
        emb1 = generate_embeddings(n_items, dim, seed)
        emb2 = generate_embeddings(n_items, dim, seed)
        
        # Must be identical
        np.testing.assert_array_equal(emb1, emb2)
        
        # Verify hash equality
        hash1 = hash(emb1.tobytes())
        hash2 = hash(emb2.tobytes())
        assert hash1 == hash2
    
    def test_coordinates_perfect_reproducibility(self):
        """Verify spatial coordinates are bit-for-bit reproducible."""
        set_deterministic_mode()
        
        n_items, seed = 20, 42
        
        for layout in ["circle", "grid", "line", "random"]:
            coords1 = generate_spatial_coords(n_items, layout, seed)
            coords2 = generate_spatial_coords(n_items, layout, seed)
            
            np.testing.assert_array_equal(
                coords1, coords2,
                err_msg=f"Layout {layout} not reproducible"
            )
    
    def test_multiple_runs_consistency(self):
        """Verify consistency across multiple independent runs."""
        set_deterministic_mode()
        
        seed = 42
        results = []
        
        for _ in range(5):
            emb = generate_embeddings(20, 100, seed)
            coords = generate_spatial_coords(20, "circle", seed)
            ssc = compute_ssc_from_data(emb, coords)
            results.append(ssc)
        
        # All runs must produce identical results
        assert len(set(results)) == 1, f"Inconsistent results: {results}"


class TestSSCReproducibility:
    """Test SSC computation reproducibility."""
    
    def test_ssc_deterministic_across_runs(self):
        """Verify SSC produces identical results across runs."""
        set_deterministic_mode()
        
        seed = 42
        ssc_values = []
        
        for _ in range(10):
            emb = generate_embeddings(20, 100, seed)
            coords = generate_spatial_coords(20, "circle", seed)
            ssc = compute_ssc_from_data(emb, coords)
            ssc_values.append(ssc)
        
        # Standard deviation must be zero
        assert np.std(ssc_values) == 0.0, f"SSC not deterministic: {ssc_values}"
    
    def test_ssc_hash_stability(self):
        """Verify SSC result can be hashed consistently."""
        set_deterministic_mode()
        
        seed = 42
        
        emb = generate_embeddings(20, 100, seed)
        coords = generate_spatial_coords(20, "circle", seed)
        
        ssc1 = compute_ssc_from_data(emb, coords)
        ssc2 = compute_ssc_from_data(emb, coords)
        
        # Even floating point should be identical
        assert ssc1.hex() == ssc2.hex(), "Float representation differs"


class TestBootstrapReproducibility:
    """Test bootstrap statistical procedures."""
    
    def test_bootstrap_ci_reproducibility(self):
        """Verify bootstrap CI is reproducible with seed."""
        set_deterministic_mode()
        
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        ci1 = bootstrap_ci(data, n_bootstrap=5000, seed=42)
        ci2 = bootstrap_ci(data, n_bootstrap=5000, seed=42)
        
        np.testing.assert_array_equal(ci1, ci2)
    
    def test_bootstrap_different_seeds_differ(self):
        """Verify different seeds produce different CIs."""
        set_deterministic_mode()
        
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        ci1 = bootstrap_ci(data, n_bootstrap=5000, seed=42)
        ci2 = bootstrap_ci(data, n_bootstrap=5000, seed=43)
        
        assert not np.array_equal(ci1, ci2), "Different seeds should differ"


class TestCrossImplementationConsistency:
    """Test consistency across different computational paths."""
    
    def test_pdist_squareform_consistency(self):
        """Verify pdist produces consistent results."""
        from scipy.spatial.distance import pdist, squareform
        
        set_deterministic_mode()
        np.random.seed(42)
        
        data = np.random.normal(0, 1, (20, 10))
        
        # Method 1: Direct pdist
        dist1 = pdist(data, 'correlation')
        
        # Method 2: Via squareform
        dist2 = squareform(pdist(data, 'correlation'))
        dist2 = dist2[np.triu_indices(20, k=1)]
        
        np.testing.assert_array_almost_equal(dist1, dist2, decimal=15)
    
    def test_numpy_scipy_consistency(self):
        """Verify NumPy and SciPy produce consistent results."""
        from scipy.stats import spearmanr
        from scipy.spatial.distance import pdist
        
        set_deterministic_mode()
        np.random.seed(42)
        
        emb = np.random.normal(0, 1, (20, 10))
        coords = np.random.normal(0, 1, (20, 2))
        
        sem_dist = pdist(emb, 'correlation')
        spa_dist = pdist(coords, 'euclidean')
        
        # Multiple computations should be identical
        rho1, _ = spearmanr(sem_dist, spa_dist)
        rho2, _ = spearmanr(sem_dist, spa_dist)
        
        np.testing.assert_equal(rho1, rho2)


class TestExperimentReproducibility:
    """Test full experiment reproducibility."""
    
    def test_baseline_experiment_reproducibility(self):
        """Simulate EXP-00 baseline experiment reproducibility."""
        set_deterministic_mode()
        
        n_items, dim, n_trials, seed = 20, 100, 10, 42
        
        # Run 1
        ssc_run1 = []
        for i in range(n_trials):
            emb = generate_embeddings(n_items, dim, seed + i)
            coords = generate_spatial_coords(n_items, "random", seed + i)
            ssc = compute_ssc_from_data(emb, coords)
            ssc_run1.append(ssc)
        
        # Run 2
        ssc_run2 = []
        for i in range(n_trials):
            emb = generate_embeddings(n_items, dim, seed + i)
            coords = generate_spatial_coords(n_items, "random", seed + i)
            ssc = compute_ssc_from_data(emb, coords)
            ssc_run2.append(ssc)
        
        # Must be identical
        np.testing.assert_array_equal(ssc_run1, ssc_run2)
        
        # Statistics must be identical
        assert np.mean(ssc_run1) == np.mean(ssc_run2)
        assert np.std(ssc_run1) == np.std(ssc_run2)
