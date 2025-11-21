"""Tests for SSC wrapper (v1 compatibility layer).

Validates:
    - Correct wrapping of v1 compute_ssc_from_data
    - Return value consistency
    - Determinism
    - v1/v2 interface equivalence
"""

import pytest
import numpy as np
from src.core_sp.ssc_wrapper import compute_ssc
from src.core.ssc_computation import compute_ssc_from_data


class TestSSCWrapper:
    """Test suite for SSC wrapper."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample embeddings and coordinates."""
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(64, 128))
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        coords = rng.uniform(-1, 1, (64, 2))
        return embeddings, coords
    
    def test_wrapper_equivalence(self, sample_data):
        """Test: Wrapper produces same result as v1 direct call."""
        embeddings, coords = sample_data
        
        # v2 wrapper
        ssc_v2 = compute_ssc(embeddings, coords)
        
        # v1 direct
        ssc_v1 = compute_ssc_from_data(
            embeddings=embeddings,
            coords=coords,
            semantic_metric='correlation',
            spatial_metric='euclidean'
        )
        
        assert ssc_v2 == ssc_v1, "Wrapper must produce identical results to v1"
    
    def test_wrapper_determinism(self, sample_data):
        """Test: Multiple calls with same data → same result."""
        embeddings, coords = sample_data
        
        ssc1 = compute_ssc(embeddings, coords)
        ssc2 = compute_ssc(embeddings, coords)
        
        assert ssc1 == ssc2, "SSC computation must be deterministic"
    
    def test_wrapper_range(self, sample_data):
        """Test: SSC in [-1, 1]."""
        embeddings, coords = sample_data
        
        ssc = compute_ssc(embeddings, coords)
        
        assert -1.0 <= ssc <= 1.0, f"SSC must be in [-1, 1], got {ssc}"
    
    def test_wrapper_natural_orthogonality(self):
        """Test: Random embeddings + random coords → SSC ≈ 0 (O-1)."""
        rng = np.random.default_rng(42)
        
        embeddings = rng.normal(size=(64, 128))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        coords = rng.uniform(-1, 1, (64, 2))
        
        ssc = compute_ssc(embeddings, coords)
        
        assert abs(ssc) < 0.15, f"Natural orthogonality: |SSC| should be small, got {ssc}"
    
    def test_wrapper_custom_metrics(self, sample_data):
        """Test: Custom distance metrics."""
        embeddings, coords = sample_data
        
        # Cosine semantic + Manhattan spatial
        ssc = compute_ssc(
            embeddings, coords,
            semantic_metric='cosine',
            spatial_metric='cityblock'
        )
        
        assert -1.0 <= ssc <= 1.0, "Custom metrics must produce valid SSC"
    
    def test_wrapper_return_pvalue(self, sample_data):
        """Test: Optional p-value return."""
        embeddings, coords = sample_data
        
        ssc, pval = compute_ssc(embeddings, coords, return_pvalue=True)
        
        assert isinstance(ssc, float), "SSC must be float"
        assert isinstance(pval, float), "p-value must be float"
        assert 0.0 <= pval <= 1.0, "p-value must be in [0, 1]"


class TestSSCEdgeCases:
    """Test edge cases."""
    
    def test_wrapper_small_n(self):
        """Test: Small sample size."""
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(10, 128))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        coords = rng.uniform(-1, 1, (10, 2))
        
        ssc = compute_ssc(embeddings, coords)
        assert -1.0 <= ssc <= 1.0
    
    def test_wrapper_mismatched_n(self):
        """Test: Mismatched N raises error."""
        embeddings = np.random.normal(size=(64, 128))
        coords = np.random.uniform(-1, 1, (32, 2))
        
        with pytest.raises(ValueError):
            compute_ssc(embeddings, coords)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
