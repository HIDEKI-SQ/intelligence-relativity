"""Tests for core measurement toolkit.

Validates the fundamental measurement instruments used across
all experiments in the Intelligence Relativity framework.
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
    compute_summary_stats,
    bootstrap_ci,
    generate_manifest
)


class TestDeterministicMode:
    """Test deterministic execution guarantees."""
    
    def test_set_deterministic_mode(self):
        """Verify deterministic mode sets environment correctly."""
        set_deterministic_mode()
        
        import os
        assert os.environ.get('OPENBLAS_NUM_THREADS') == '1'
        assert os.environ.get('MKL_NUM_THREADS') == '1'
        assert os.environ.get('OMP_NUM_THREADS') == '1'
    
    def test_verify_environment_creates_file(self, tmp_path):
        """Verify environment verification creates output file."""
        env_file = tmp_path / "env.txt"
        verify_environment(env_file)
        
        assert env_file.exists()
        
        with open(env_file) as f:
            env_data = json.load(f)
        
        assert 'python' in env_data
        assert 'numpy' in env_data
        assert 'scipy' in env_data


class TestGenerators:
    """Test data generation utilities."""
    
    def test_generate_embeddings_shape(self):
        """Verify embedding generation produces correct shape."""
        n_items = 20
        dim = 100
        seed = 42
        
        embeddings = generate_embeddings(n_items, dim, seed)
        
        assert embeddings.shape == (n_items, dim)
        assert embeddings.dtype == np.float64
    
    def test_generate_embeddings_reproducibility(self):
        """Verify embeddings are reproducible with same seed."""
        n_items = 20
        dim = 100
        seed = 42
        
        emb1 = generate_embeddings(n_items, dim, seed)
        emb2 = generate_embeddings(n_items, dim, seed)
        
        np.testing.assert_array_equal(emb1, emb2)
    
    def test_generate_embeddings_different_seeds(self):
        """Verify different seeds produce different embeddings."""
        n_items = 20
        dim = 100
        
        emb1 = generate_embeddings(n_items, dim, seed=42)
        emb2 = generate_embeddings(n_items, dim, seed=43)
        
        assert not np.array_equal(emb1, emb2)
    
    @pytest.mark.parametrize("layout", ["circle", "grid", "line", "random"])
    def test_generate_spatial_coords_layouts(self, layout):
        """Verify all spatial layouts generate correct shapes."""
        n_items = 20
        seed = 42
        
        coords = generate_spatial_coords(n_items, layout, seed)
        
        assert coords.shape[0] == n_items
        assert coords.shape[1] in [2, 3]  # 2D or 3D
    
    def test_generate_spatial_coords_reproducibility(self):
        """Verify spatial coordinates are reproducible."""
        n_items = 20
        seed = 42
        
        coords1 = generate_spatial_coords(n_items, "circle", seed)
        coords2 = generate_spatial_coords(n_items, "circle", seed)
        
        np.testing.assert_array_equal(coords1, coords2)


class TestStatistics:
    """Test statistical utilities."""
    
    def test_compute_summary_stats_structure(self):
        """Verify summary statistics returns correct structure."""
        data = np.random.normal(0, 1, 1000)
        
        stats = compute_summary_stats(data)
        
        required_keys = ['mean', 'std', 'median', 'min', 'max', 'n']
        for key in required_keys:
            assert key in stats
    
    def test_compute_summary_stats_values(self):
        """Verify summary statistics computes correct values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        stats = compute_summary_stats(data)
        
        assert stats['mean'] == pytest.approx(3.0)
        assert stats['median'] == pytest.approx(3.0)
        assert stats['min'] == pytest.approx(1.0)
        assert stats['max'] == pytest.approx(5.0)
        assert stats['n'] == 5
    
    def test_bootstrap_ci_reproducibility(self):
        """Verify bootstrap CI is reproducible with same seed."""
        data = np.random.normal(0, 1, 100)
        
        ci1 = bootstrap_ci(data, n_bootstrap=1000, seed=42)
        ci2 = bootstrap_ci(data, n_bootstrap=1000, seed=42)
        
        np.testing.assert_array_almost_equal(ci1, ci2)
    
    def test_bootstrap_ci_range(self):
        """Verify bootstrap CI returns reasonable range."""
        data = np.random.normal(0, 1, 1000)
        
        ci = bootstrap_ci(data, n_bootstrap=5000, seed=42)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower < Upper
        assert ci[0] < np.mean(data) < ci[1]  # Mean within CI


class TestManifestGeneration:
    """Test manifest generation for reproducibility."""
    
    def test_generate_manifest_creates_file(self, tmp_path):
        """Verify manifest generation creates output file."""
        # Create test files
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        manifest_file = tmp_path / "manifest.json"
        generate_manifest(tmp_path, manifest_file)
        
        assert manifest_file.exists()
        
        with open(manifest_file) as f:
            manifest = json.load(f)
        
        assert "test.txt" in manifest
        assert isinstance(manifest["test.txt"], str)  # SHA256 hash
        assert len(manifest["test.txt"]) == 64  # SHA256 length
    
    def test_generate_manifest_hash_stability(self, tmp_path):
        """Verify manifest generates same hash for same content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("stable content")
        
        manifest_file1 = tmp_path / "manifest1.json"
        generate_manifest(tmp_path, manifest_file1)
        
        manifest_file2 = tmp_path / "manifest2.json"
        generate_manifest(tmp_path, manifest_file2)
        
        with open(manifest_file1) as f:
            manifest1 = json.load(f)
        with open(manifest_file2) as f:
            manifest2 = json.load(f)
        
        assert manifest1["test.txt"] == manifest2["test.txt"]


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline_reproducibility(self):
        """Verify complete pipeline is reproducible."""
        set_deterministic_mode()
        
        n_items = 20
        dim = 100
        seed = 42
        
        # Run 1
        emb1 = generate_embeddings(n_items, dim, seed)
        coords1 = generate_spatial_coords(n_items, "circle", seed)
        
        # Run 2
        emb2 = generate_embeddings(n_items, dim, seed)
        coords2 = generate_spatial_coords(n_items, "circle", seed)
        
        np.testing.assert_array_equal(emb1, emb2)
        np.testing.assert_array_equal(coords1, coords2)
    
    def test_statistical_pipeline(self):
        """Verify statistical analysis pipeline."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        stats = compute_summary_stats(data)
        ci = bootstrap_ci(data, n_bootstrap=1000, seed=42)
        
        # Verify consistency
        assert ci[0] < stats['mean'] < ci[1]
        assert stats['n'] == len(data)
