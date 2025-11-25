"""Tests for value gate mechanism.

Validates:
    - λ=0 → returns base_coords (Structure preserved)
    - λ=1 → semantic alignment (Meaning dominant)
    - Determinism with seed
    - Monotonic SSC increase with λ
"""

import pytest
import numpy as np
from src.core_sp.value_gate import apply_value_gate
from src.core_sp.ssc_wrapper import compute_ssc
from src.core_sp.generators import generate_semantic_embeddings


class TestValueGate:
    """Test suite for value gate mechanism."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample embeddings and base coordinates."""
        rng = np.random.default_rng(42)
        embeddings = generate_semantic_embeddings(20, 100, rng)
        base_coords = rng.uniform(-1, 1, (20, 2))
        return embeddings, base_coords
    
    def test_value_gate_determinism(self, sample_data):
        """Test: Same seed → same arrangement."""
        embeddings, base_coords = sample_data
        
        coords1 = apply_value_gate(base_coords, embeddings, lam=0.5, seed=42)
        coords2 = apply_value_gate(base_coords, embeddings, lam=0.5, seed=42)
        
        assert np.allclose(coords1, coords2), "Value gate must be deterministic"
    
    def test_value_gate_different_seeds(self, sample_data):
        """Test: Different seeds → different arrangements (only relevant when λ > 0)."""
        embeddings, base_coords = sample_data
        
        # Use λ=1.0 to ensure the difference comes from PCA projection (which uses the seed)
        # At λ=0, it returns base_coords so seed doesn't matter.
        lam_val = 1.0 
        
        coords1 = apply_value_gate(base_coords, embeddings, lam=lam_val, seed=42)
        coords2 = apply_value_gate(base_coords, embeddings, lam=lam_val, seed=99)
        
        assert not np.allclose(coords1, coords2), "Different seeds must produce different results when λ > 0"
    
    def test_value_gate_shape(self, sample_data):
        """Test: Output shape matches input."""
        embeddings, base_coords = sample_data
        
        coords_gated = apply_value_gate(base_coords, embeddings, lam=0.5, seed=42)
        
        assert coords_gated.shape == (20, 2), f"Expected (20, 2), got {coords_gated.shape}"
    
    def test_value_gate_lambda_range(self, sample_data):
        """Test: λ ∈ [0, 1] produces valid coordinates."""
        embeddings, base_coords = sample_data
        
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            coords = apply_value_gate(base_coords, embeddings, lam=lam, seed=42)
            assert coords.shape == (20, 2)
            assert np.all(np.isfinite(coords)), f"Invalid coords at λ={lam}"
    
    def test_value_gate_lambda0_randomness(self, sample_data):
        """Test: λ=0 → returns base_coords exactly (Structure Preserved)."""
        embeddings, base_coords = sample_data
        rng = np.random.default_rng(42)
        
        # Even with different embeddings, λ=0 should return the original base_coords
        embeddings2 = generate_semantic_embeddings(20, 100, rng)
        
        coords1 = apply_value_gate(base_coords, embeddings, lam=0.0, seed=42)
        coords2 = apply_value_gate(base_coords, embeddings2, lam=0.0, seed=42)
        
        assert np.allclose(coords1, base_coords), "λ=0 must return base_coords exactly"
        assert np.allclose(coords2, base_coords), "λ=0 must return base_coords exactly"
    
    def test_value_gate_monotonic_ssc(self):
        """Test: λ ↑ → SSC ↑ (monotonic coupling, O-4)."""
        rng = np.random.default_rng(42)
        embeddings = generate_semantic_embeddings(50, 100, rng)
        # Start with random base coords (low SSC)
        base_coords = rng.uniform(-1, 1, (50, 2))
        
        lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        sscs = []
        
        for lam in lambdas:
            coords = apply_value_gate(base_coords, embeddings, lam=lam, seed=42)
            ssc = compute_ssc(embeddings, coords)
            sscs.append(ssc)
        
        # Check general increasing trend (allowing some noise)
        # Compare first half vs second half
        ssc_low = np.mean(sscs[:3])  # λ=[0.0, 0.2, 0.4]
        ssc_high = np.mean(sscs[3:])  # λ=[0.6, 0.8, 1.0]
        
        assert ssc_high > ssc_low, f"SSC should increase with λ: {sscs}"
    
    # REMOVED: test_value_gate_circle_layout (Deprecated behavior)
    # REMOVED: test_value_gate_unsupported_layout (Deprecated check)


class TestValueGateEdgeCases:
    """Test edge cases."""
    
    def test_value_gate_small_n(self):
        """Test: Works with small N."""
        rng = np.random.default_rng(42)
        embeddings = generate_semantic_embeddings(5, 50, rng)
        base_coords = rng.uniform(-1, 1, (5, 2))
        
        coords = apply_value_gate(base_coords, embeddings, lam=0.5, seed=42)
        assert coords.shape == (5, 2)
    
    def test_value_gate_large_dim(self):
        """Test: Works with large embedding dimension."""
        rng = np.random.default_rng(42)
        embeddings = generate_semantic_embeddings(10, 512, rng)
        base_coords = rng.uniform(-1, 1, (10, 2))
        
        coords = apply_value_gate(base_coords, embeddings, lam=0.5, seed=42)
        assert coords.shape == (10, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
