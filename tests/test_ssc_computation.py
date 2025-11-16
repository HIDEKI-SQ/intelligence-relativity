"""Tests for SSC (Semantic-Spatial Correlation) computation.

These tests validate the core measurement instrument used throughout
the Intelligence Relativity framework. The SSC measurement is the
foundation of all O-1 through O-4 observations.
"""

import pytest
import numpy as np
from scipy.spatial.distance import pdist

from src.core import compute_ssc, compute_ssc_from_data


class TestSSCComputation:
    """Test SSC computation from distance vectors."""
    
    def test_compute_ssc_perfect_correlation(self):
        """Verify SSC detects perfect positive correlation."""
        # Create perfectly correlated distances
        n_items = 10
        
        # Same ordering in both spaces
        order = np.arange(n_items).astype(float)
        
        # Semantic: ordered embeddings
        emb = order.reshape(-1, 1)
        sem_dist = pdist(emb, 'euclidean')
        
        # Spatial: same ordering
        coords = np.column_stack([order, np.zeros(n_items)])
        spa_dist = pdist(coords, 'euclidean')
        
        ssc = compute_ssc(sem_dist, spa_dist)
        
        # Should be high positive correlation
        assert ssc > 0.9, f"Expected SSC > 0.9, got {ssc}"
    
    def test_compute_ssc_zero_correlation(self):
        """Verify SSC detects zero correlation (O-1)."""
        np.random.seed(42)
        n_items = 20
        
        # Random independent structures
        emb = np.random.normal(0, 1, (n_items, 10))
        coords = np.random.normal(0, 1, (n_items, 2))
        
        sem_dist = pdist(emb, 'correlation')
        spa_dist = pdist(coords, 'euclidean')
        
        ssc = compute_ssc(sem_dist, spa_dist)
        
        # Should be close to zero
        assert abs(ssc) < 0.15, f"Expected |SSC| < 0.15, got {ssc}"
    
    def test_compute_ssc_input_validation(self):
        """Verify SSC validates input shapes."""
        # Mismatched shapes
        sem_dist = np.random.normal(0, 1, 10)
        spa_dist = np.random.normal(0, 1, 15)
        
        with pytest.raises(ValueError, match="must have same shape"):
            compute_ssc(sem_dist, spa_dist)
    
    def test_compute_ssc_dimension_validation(self):
        """Verify SSC requires 1D condensed vectors."""
        # 2D array instead of condensed vector
        sem_dist = np.random.normal(0, 1, (10, 10))
        spa_dist = np.random.normal(0, 1, (10, 10))
        
        with pytest.raises(ValueError, match="1D condensed vectors"):
            compute_ssc(sem_dist, spa_dist)
    
    def test_compute_ssc_return_pvalue(self):
        """Verify SSC can return p-value."""
        np.random.seed(42)
        n_items = 20
        
        emb = np.random.normal(0, 1, (n_items, 10))
        coords = np.random.normal(0, 1, (n_items, 2))
        
        sem_dist = pdist(emb, 'correlation')
        spa_dist = pdist(coords, 'euclidean')
        
        ssc, pval = compute_ssc(sem_dist, spa_dist, return_pvalue=True)
        
        assert isinstance(ssc, float)
        assert isinstance(pval, float)
        assert 0 <= pval <= 1
    
    def test_compute_ssc_handles_constant_vectors(self):
        """Verify SSC handles constant distance vectors gracefully."""
        # Constant distances (all same)
        sem_dist = np.ones(45)  # 10 choose 2 = 45
        spa_dist = np.ones(45)
        
        ssc = compute_ssc(sem_dist, spa_dist)
        
        # Should return 0 for undefined correlation
        assert ssc == 0.0


class TestSSCFromData:
    """Test SSC computation directly from data."""
    
    def test_compute_ssc_from_data_basic(self):
        """Verify SSC can be computed from raw data."""
        np.random.seed(42)
        n_items = 20
        
        emb = np.random.normal(0, 1, (n_items, 10))
        coords = np.random.normal(0, 1, (n_items, 2))
        
        ssc = compute_ssc_from_data(emb, coords)
        
        assert isinstance(ssc, float)
        assert -1 <= ssc <= 1
    
    def test_compute_ssc_from_data_reproducibility(self):
        """Verify SSC from data is reproducible."""
        np.random.seed(42)
        n_items = 20
        
        emb = np.random.normal(0, 1, (n_items, 10))
        coords = np.random.normal(0, 1, (n_items, 2))
        
        ssc1 = compute_ssc_from_data(emb, coords)
        ssc2 = compute_ssc_from_data(emb, coords)
        
        assert ssc1 == ssc2
    
    @pytest.mark.parametrize("semantic_metric", ["correlation", "euclidean", "cosine"])
    def test_compute_ssc_from_data_metrics(self, semantic_metric):
        """Verify SSC works with different semantic metrics."""
        np.random.seed(42)
        n_items = 20
        
        emb = np.random.normal(0, 1, (n_items, 10))
        coords = np.random.normal(0, 1, (n_items, 2))
        
        ssc = compute_ssc_from_data(
            emb, coords,
            semantic_metric=semantic_metric,
            spatial_metric='euclidean'
        )
        
        assert isinstance(ssc, float)
        assert -1 <= ssc <= 1
    
    def test_compute_ssc_from_data_with_pvalue(self):
        """Verify SSC from data can return p-value."""
        np.random.seed(42)
        n_items = 20
        
        emb = np.random.normal(0, 1, (n_items, 10))
        coords = np.random.normal(0, 1, (n_items, 2))
        
        ssc, pval = compute_ssc_from_data(emb, coords, return_pvalue=True)
        
        assert isinstance(ssc, float)
        assert isinstance(pval, float)
        assert 0 <= pval <= 1


class TestSSCReproducibility:
    """Test SSC reproducibility guarantees."""
    
    def test_ssc_cross_implementation_consistency(self):
        """Verify SSC is consistent across different computation paths."""
        np.random.seed(42)
        n_items = 20
        
        emb = np.random.normal(0, 1, (n_items, 10))
        coords = np.random.normal(0, 1, (n_items, 2))
        
        # Method 1: Manual distance computation
        sem_dist = pdist(emb, 'correlation')
        spa_dist = pdist(coords, 'euclidean')
        ssc1 = compute_ssc(sem_dist, spa_dist)
        
        # Method 2: Direct from data
        ssc2 = compute_ssc_from_data(
            emb, coords,
            semantic_metric='correlation',
            spatial_metric='euclidean'
        )
        
        np.testing.assert_almost_equal(ssc1, ssc2, decimal=10)
    
    def test_ssc_numerical_stability(self):
        """Verify SSC is numerically stable."""
        np.random.seed(42)
        n_items = 50
        
        emb = np.random.normal(0, 1, (n_items, 100))
        coords = np.random.normal(0, 1, (n_items, 2))
        
        # Compute multiple times
        ssc_values = []
        for _ in range(10):
            sem_dist = pdist(emb, 'correlation')
            spa_dist = pdist(coords, 'euclidean')
            ssc = compute_ssc(sem_dist, spa_dist)
            ssc_values.append(ssc)
        
        # Should be identical
        assert len(set(ssc_values)) == 1, "SSC not numerically stable"


class TestSSCObservations:
    """Test SSC behavior under different observation scenarios."""
    
    def test_ssc_o1_natural_orthogonality(self):
        """Test O-1: Natural orthogonality at λ=0."""
        np.random.seed(42)
        n_trials = 100
        ssc_values = []
        
        for i in range(n_trials):
            emb = np.random.normal(0, 1, (20, 10))
            coords = np.random.normal(0, 1, (20, 2))
            
            ssc = compute_ssc_from_data(emb, coords)
            ssc_values.append(ssc)
        
        mean_ssc = np.mean(ssc_values)
        
        # O-1: Should be close to zero
        assert abs(mean_ssc) < 0.10, f"O-1 violation: mean SSC = {mean_ssc}"
    
    def test_ssc_sensitivity_to_correlation(self):
        """Verify SSC is sensitive to induced correlation."""
        np.random.seed(42)
        n_items = 30
        
        # Create correlated structures
        order = np.arange(n_items).astype(float)
        
        # Embeddings with ordering
        emb = np.column_stack([
            order / n_items,
            order / n_items + np.random.normal(0, 0.01, n_items),
            order / n_items + np.random.normal(0, 0.01, n_items),
        ])
        
        # Coords with same ordering
        coords = np.column_stack([
            order,
            np.zeros(n_items)
        ])
        
        ssc = compute_ssc_from_data(emb, coords, semantic_metric='euclidean')
        
        # Should detect correlation
        assert ssc > 0.5, f"Failed to detect correlation: SSC = {ssc}"
