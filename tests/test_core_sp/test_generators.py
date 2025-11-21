"""Tests for semantic generators.

Validates:
    - Determinism
    - Unit normalization
    - Shape preservation
    - Noise properties
"""

import pytest
import numpy as np
from src.core_sp.generators import generate_semantic_embeddings, add_semantic_noise


class TestGenerators:
    """Test suite for semantic generators."""
    
    def test_generate_determinism(self):
        """Test: Same seed → same embeddings."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        emb1 = generate_semantic_embeddings(64, 128, rng1)
        emb2 = generate_semantic_embeddings(64, 128, rng2)
        
        assert np.allclose(emb1, emb2), "Generation must be deterministic"
    
    def test_generate_shape(self):
        """Test: Correct output shape."""
        rng = np.random.default_rng(42)
        emb = generate_semantic_embeddings(n_items=100, dim=256, rng=rng)
        
        assert emb.shape == (100, 256), f"Expected (100, 256), got {emb.shape}"
    
    def test_generate_unit_norm(self):
        """Test: All embeddings have unit norm."""
        rng = np.random.default_rng(42)
        emb = generate_semantic_embeddings(64, 128, rng)
        
        norms = np.linalg.norm(emb, axis=1)
        
        assert np.allclose(norms, 1.0, atol=1e-10), "All embeddings must have unit norm"
    
    def test_generate_different_seeds(self):
        """Test: Different seeds → different embeddings."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        
        emb1 = generate_semantic_embeddings(64, 128, rng1)
        emb2 = generate_semantic_embeddings(64, 128, rng2)
        
        assert not np.allclose(emb1, emb2), "Different seeds must produce different embeddings"
    
    def test_generate_finite(self):
        """Test: All values are finite."""
        rng = np.random.default_rng(42)
        emb = generate_semantic_embeddings(64, 128, rng)
        
        assert np.all(np.isfinite(emb)), "All values must be finite"
    
    def test_add_noise_determinism(self):
        """Test: Same seed → same noise."""
        rng_gen = np.random.default_rng(42)
        emb = generate_semantic_embeddings(64, 128, rng_gen)
        
        rng1 = np.random.default_rng(100)
        rng2 = np.random.default_rng(100)
        
        emb_noisy1 = add_semantic_noise(emb, rng1, sigma=0.1)
        emb_noisy2 = add_semantic_noise(emb, rng2, sigma=0.1)
        
        assert np.allclose(emb_noisy1, emb_noisy2), "Noise addition must be deterministic"
    
    def test_add_noise_sigma0(self):
        """Test: σ=0 → no change."""
        rng_gen = np.random.default_rng(42)
        emb = generate_semantic_embeddings(64, 128, rng_gen)
        
        rng_noise = np.random.default_rng(100)
        emb_noisy = add_semantic_noise(emb, rng_noise, sigma=0.0)
        
        assert np.allclose(emb_noisy, emb, atol=1e-10), "σ=0 should add no noise"
    
    def test_add_noise_unit_norm(self):
        """Test: Noisy embeddings remain unit norm."""
        rng_gen = np.random.default_rng(42)
        emb = generate_semantic_embeddings(64, 128, rng_gen)
        
        rng_noise = np.random.default_rng(100)
        emb_noisy = add_semantic_noise(emb, rng_noise, sigma=0.3)
        
        norms = np.linalg.norm(emb_noisy, axis=1)
        
        assert np.allclose(norms, 1.0, atol=1e-10), "Noisy embeddings must have unit norm"
    
    def test_add_noise_shape(self):
        """Test: Noise preserves shape."""
        rng_gen = np.random.default_rng(42)
        emb = generate_semantic_embeddings(64, 128, rng_gen)
        
        rng_noise = np.random.default_rng(100)
        emb_noisy = add_semantic_noise(emb, rng_noise, sigma=0.2)
        
        assert emb_noisy.shape == emb.shape, "Shape must be preserved"
    
    def test_add_noise_finite(self):
        """Test: Noisy embeddings are finite."""
        rng_gen = np.random.default_rng(42)
        emb = generate_semantic_embeddings(64, 128, rng_gen)
        
        rng_noise = np.random.default_rng(100)
        emb_noisy = add_semantic_noise(emb, rng_noise, sigma=0.5)
        
        assert np.all(np.isfinite(emb_noisy)), "All values must be finite"
    
    def test_add_noise_different_sigma(self):
        """Test: Different σ → different noise levels."""
        rng_gen = np.random.default_rng(42)
        emb = generate_semantic_embeddings(64, 128, rng_gen)
        
        rng_noise = np.random.default_rng(100)
        emb_noisy_small = add_semantic_noise(emb.copy(), rng_noise, sigma=0.1)
        
        rng_noise = np.random.default_rng(100)
        emb_noisy_large = add_semantic_noise(emb.copy(), rng_noise, sigma=0.5)
        
        # Larger sigma should produce larger deviations
        dev_small = np.linalg.norm(emb - emb_noisy_small)
        dev_large = np.linalg.norm(emb - emb_noisy_large)
        
        assert dev_large > dev_small, "Larger σ should produce larger deviations"


class TestGeneratorEdgeCases:
    """Test edge cases."""
    
    def test_generate_small_dim(self):
        """Test: Small dimension."""
        rng = np.random.default_rng(42)
        emb = generate_semantic_embeddings(10, 2, rng)
        
        assert emb.shape == (10, 2)
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)
    
    def test_generate_single_item(self):
        """Test: Single item."""
        rng = np.random.default_rng(42)
        emb = generate_semantic_embeddings(1, 128, rng)
        
        assert emb.shape == (1, 128)
        assert np.allclose(np.linalg.norm(emb), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
