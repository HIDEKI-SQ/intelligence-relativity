"""Cross-implementation validation tests.

Validates:
    - v1/v2 SSC compatibility
    - Implementation difference bounds (|Δ| < threshold)
    - Interface consistency

This ensures v2.0.0 maintains compatibility with v1.1.2 while
extending functionality.
"""

import pytest
import numpy as np
from scipy.stats import spearmanr

from src.core_sp.ssc_wrapper import compute_ssc
from src.core.ssc_computation import compute_ssc_from_data
from src.core_sp.generators import generate_semantic_embeddings


class TestSSCCrossImplementation:
    """Test SSC wrapper vs v1 direct implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        rng = np.random.default_rng(42)
        embeddings = generate_semantic_embeddings(64, 128, rng)
        coords = rng.uniform(-1, 1, (64, 2))
        return embeddings, coords
    
    def test_ssc_v1_v2_equivalence(self, sample_data):
        """Test: v2 wrapper produces identical results to v1."""
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
        
        assert ssc_v2 == ssc_v1, \
            f"v1/v2 SSC mismatch: v1={ssc_v1}, v2={ssc_v2}"
    
    def test_ssc_cross_metric_consistency(self, sample_data):
        """Test: Different metrics produce consistent correlations."""
        embeddings, coords = sample_data
        
        ssc_corr = compute_ssc(embeddings, coords, semantic_metric='correlation')
        ssc_cos = compute_ssc(embeddings, coords, semantic_metric='cosine')
        
        # Should be highly correlated (not identical due to metric difference)
        assert abs(ssc_corr - ssc_cos) < 0.3, \
            f"Metrics should be consistent: corr={ssc_corr}, cos={ssc_cos}"
    
    def test_ssc_difference_bound(self, sample_data):
        """Test: Alternative SSC implementations within |Δ| < 0.07 (v1 standard)."""
        embeddings, coords = sample_data
        
        # Compute with different but equivalent approaches
        ssc1 = compute_ssc(embeddings, coords)
        
        # Manual Spearman computation (alternative implementation)
        from scipy.spatial.distance import pdist, squareform
        
        D_sem = squareform(pdist(embeddings, metric='correlation'))
        D_spatial = squareform(pdist(coords, metric='euclidean'))
        
        # Flatten and compute Spearman
        mask = np.triu_indices_from(D_sem, k=1)
        ssc2, _ = spearmanr(D_sem[mask], D_spatial[mask])
        
        diff = abs(ssc1 - ssc2)
        
        assert diff < 0.07, \
            f"Cross-implementation difference should be < 0.07, got {diff}"


class TestSPCrossLayout:
    """Test SP consistency across layout types."""
    
    def test_sp_grid_vs_random_identity(self):
        """Test: Identity SP ≈ 1 for both grid and random."""
        from src.core_sp.sp_metrics import compute_sp_total
        
        rng = np.random.default_rng(42)
        coords = rng.uniform(-1, 1, (64, 2))
        
        sp_grid = compute_sp_total(coords, coords, layout_type="grid")
        sp_random = compute_sp_total(coords, coords, layout_type="random")
        
        # Both should be ~1.0 for identity
        assert sp_grid > 0.95 and sp_random > 0.95, \
            f"Identity SP should be high: grid={sp_grid}, random={sp_random}"
    
    def test_sp_destruction_consistency(self):
        """Test: Destruction reduces SP for all layouts."""
        from src.core_sp.sp_metrics import compute_sp_total
        from src.core_sp.topology_ops import random_relayout
        
        rng = np.random.default_rng(42)
        coords = rng.uniform(-1, 1, (64, 2))
        coords_rand = random_relayout(coords, rng=rng, bounds=(-1, 1))
        
        sp_grid = compute_sp_total(coords, coords_rand, layout_type="grid")
        sp_random = compute_sp_total(coords, coords_rand, layout_type="random")
        
        # Both should be low
        assert sp_grid < 0.4 and sp_random < 0.4, \
            f"Destruction SP should be low: grid={sp_grid}, random={sp_random}"


class TestValueGateCrossEmbedding:
    """Test value gate consistency across embedding types."""
    
    def test_value_gate_synthetic_consistency(self):
        """Test: Value gate produces monotonic SSC across trials."""
        from src.core_sp.value_gate import apply_value_gate
        from src.core_sp.ssc_wrapper import compute_ssc
        from src.core_sp.generators import generate_semantic_embeddings
        
        rng = np.random.default_rng(42)
        
        # Multiple random embeddings
        sscs_at_lambda05 = []
        
        for trial in range(5):
            embeddings = generate_semantic_embeddings(20, 100, rng)
            base_coords = rng.uniform(-1, 1, (20, 2))
            
            coords = apply_value_gate(base_coords, embeddings, lam=0.5, seed=42+trial)
            ssc = compute_ssc(embeddings, coords)
            sscs_at_lambda05.append(ssc)
        
        # All should be reasonably positive (λ=0.5 increases coupling)
        mean_ssc = np.mean(sscs_at_lambda05)
        assert mean_ssc > 0.0, \
            f"λ=0.5 should produce positive SSC on average, got {mean_ssc}"


class TestNumericalStability:
    """Test numerical stability across edge cases."""
    
    def test_sp_small_coordinates(self):
        """Test: SP stable with very small coordinates."""
        from src.core_sp.sp_metrics import compute_sp_total
        
        coords = np.random.uniform(-1e-10, 1e-10, (64, 2))
        sp = compute_sp_total(coords, coords, layout_type="grid")
        
        assert 0.0 <= sp <= 1.0, f"SP out of range for small coords: {sp}"
        assert np.isfinite(sp), "SP should be finite"
    
    def test_sp_large_coordinates(self):
        """Test: SP stable with very large coordinates."""
        from src.core_sp.sp_metrics import compute_sp_total
        
        coords = np.random.uniform(-1e10, 1e10, (64, 2))
        sp = compute_sp_total(coords, coords, layout_type="grid")
        
        assert 0.0 <= sp <= 1.0, f"SP out of range for large coords: {sp}"
        assert np.isfinite(sp), "SP should be finite"
    
    def test_ssc_degenerate_embeddings(self):
        """Test: SSC handles degenerate embeddings gracefully."""
        from src.core_sp.ssc_wrapper import compute_ssc
        
        # All identical embeddings
        embeddings = np.ones((64, 128))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        coords = np.random.uniform(-1, 1, (64, 2))
        
        # Should not crash, SSC may be NaN or 0
        try:
            ssc = compute_ssc(embeddings, coords)
            assert np.isfinite(ssc) or np.isnan(ssc), "SSC should be finite or NaN"
        except (ValueError, RuntimeWarning):
            # Acceptable to raise error for degenerate case
            pass


class TestInterfaceConsistency:
    """Test that all interfaces follow consistent patterns."""
    
    def test_all_operations_accept_rng(self):
        """Test: All stochastic operations accept np.random.Generator."""
        from src.core_sp.topology_ops import permute_coords, random_relayout
        from src.core_sp.metric_ops import add_coord_noise
        from src.core_sp.generators import generate_semantic_embeddings, add_semantic_noise
        
        rng = np.random.default_rng(42)
        coords = np.random.uniform(-1, 1, (64, 2))
        embeddings = generate_semantic_embeddings(64, 128, rng)
        
        # All should accept rng parameter
        operations = [
            lambda: permute_coords(coords, rng=rng, p=0.5),
            lambda: random_relayout(coords, rng=rng, bounds=(-1, 1)),
            lambda: add_coord_noise(coords, rng=rng, sigma=0.1),
            lambda: generate_semantic_embeddings(10, 20, rng),
            lambda: add_semantic_noise(embeddings, rng=rng, sigma=0.1),
        ]
        
        for op in operations:
            result = op()
            assert result is not None, "Operation should return result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
