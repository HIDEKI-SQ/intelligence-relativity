"""Tests for value_gate module."""
import numpy as np
import pytest
from src.core_sp.value_gate import apply_value_gate
from src.core_sp.generators import generate_semantic_embeddings

class TestValueGate:
    """Test apply_value_gate function."""

    def test_value_gate_determinism(self):
        """Test that results are deterministic with fixed seed."""
        n_items = 20
        dim = 10
        rng = np.random.default_rng(42)
        base_coords = rng.uniform(-1, 1, (n_items, 2))
        embeddings = generate_semantic_embeddings(n_items, dim, rng)
        
        # Run twice with same parameters
        coords1 = apply_value_gate(base_coords, embeddings, lam=0.5, seed=123)
        coords2 = apply_value_gate(base_coords, embeddings, lam=0.5, seed=123)
        
        assert np.allclose(coords1, coords2), "Deterministic execution failed"

    def test_value_gate_different_seeds(self):
        """Test that different seeds produce different results (when lambda > 0)."""
        n_items = 20
        dim = 10
        rng = np.random.default_rng(42)
        base_coords = rng.uniform(-1, 1, (n_items, 2))
        embeddings = generate_semantic_embeddings(n_items, dim, rng)
        
        # Run with different seeds, lambda > 0 to ensure PCA diff matters
        coords1 = apply_value_gate(base_coords, embeddings, lam=0.5, seed=123)
        coords2 = apply_value_gate(base_coords, embeddings, lam=0.5, seed=456)
        
        assert not np.allclose(coords1, coords2), "Different seeds must produce different results"

    def test_value_gate_shape(self):
        """Test output shape matches input base_coords."""
        n_items = 15
        dim = 8
        rng = np.random.default_rng(42)
        base_coords = rng.uniform(-1, 1, (n_items, 2))
        embeddings = generate_semantic_embeddings(n_items, dim, rng)
        
        coords = apply_value_gate(base_coords, embeddings, lam=0.5)
        assert coords.shape == base_coords.shape

    def test_value_gate_lambda_range(self):
        """Test lambda=0 returns base_coords and lambda=1 returns something else."""
        n_items = 20
        dim = 10
        rng = np.random.default_rng(42)
        base_coords = rng.uniform(-1, 1, (n_items, 2))
        embeddings = generate_semantic_embeddings(n_items, dim, rng)
        
        # lambda=0 -> exactly base_coords
        coords0 = apply_value_gate(base_coords, embeddings, lam=0.0)
        assert np.allclose(coords0, base_coords), "lambda=0 should return base_coords exactly"
        
        # lambda=1 -> purely semantic projection
        coords1 = apply_value_gate(base_coords, embeddings, lam=1.0)
        assert not np.allclose(coords1, base_coords), "lambda=1 should differ from base_coords"

    def test_value_gate_monotonic_ssc(self):
        """
        Test that SSC tends to increase with lambda.
        Note: This is a stochastic property, so we check a clear trend 
        or just that 1.0 > 0.0 is likely.
        For a unit test, checking determinism/shape is more important,
        but let's check end-points.
        """
        from src.core_sp.ssc_wrapper import compute_ssc
        
        n_items = 50
        dim = 32
        rng = np.random.default_rng(999)
        base_coords = rng.uniform(-1, 1, (n_items, 2)) # Random layout -> SSC~0
        embeddings = generate_semantic_embeddings(n_items, dim, rng)
        
        coords0 = apply_value_gate(base_coords, embeddings, lam=0.0, seed=42)
        ssc0 = compute_ssc(embeddings, coords0)
        
        coords1 = apply_value_gate(base_coords, embeddings, lam=1.0, seed=42)
        ssc1 = compute_ssc(embeddings, coords1)
        
        # With lambda=1, layout reflects semantic structure -> SSC should be higher
        assert ssc1 > ssc0, f"SSC should increase: {ssc0} -> {ssc1}"

class TestValueGateEdgeCases:
    """Edge cases for value gate."""
    
    def test_value_gate_small_n(self):
        n_items = 3 # Minimal for correlation
        dim = 5
        rng = np.random.default_rng(42)
        base_coords = rng.uniform(0, 1, (n_items, 2))
        embeddings = generate_semantic_embeddings(n_items, dim, rng)
        
        coords = apply_value_gate(base_coords, embeddings, lam=0.5)
        assert coords.shape == (3, 2)

    def test_value_gate_large_dim(self):
        n_items = 10
        dim = 512 # Large embedding dim
        rng = np.random.default_rng(42)
        base_coords = rng.uniform(0, 1, (n_items, 2))
        embeddings = generate_semantic_embeddings(n_items, dim, rng)
        
        coords = apply_value_gate(base_coords, embeddings, lam=0.5)
        assert coords.shape == (10, 2)
