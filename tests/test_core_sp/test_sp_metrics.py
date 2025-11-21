"""Tests for SP metrics computation.

Validates:
    - Deterministic computation (same input → same output)
    - Boundary conditions (identity, destruction)
    - Component independence
    - Multi-layout robustness
"""

import pytest
import numpy as np
from src.core_sp.sp_metrics import (
    compute_sp_total,
    compute_sp_components,
    compute_sp_adj,
    compute_sp_ord,
    compute_sp_clu,
    compute_knn_graph,
    jaccard_similarity_bool,
)


class TestSPMetrics:
    """Test suite for SP metrics."""
    
    @pytest.fixture
    def grid_coords(self):
        """8x8 grid layout."""
        xs = np.linspace(-1, 1, 8)
        ys = np.linspace(-1, 1, 8)
        xv, yv = np.meshgrid(xs, ys)
        return np.stack([xv.ravel(), yv.ravel()], axis=1)
    
    def test_sp_identity(self, grid_coords):
        """Test: Identity transformation → SP = 1.0."""
        sp = compute_sp_total(grid_coords, grid_coords, layout_type="grid")
        assert sp == pytest.approx(1.0, abs=1e-6), "Identity should preserve structure perfectly"
    
    def test_sp_determinism(self, grid_coords):
        """Test: Same inputs → same outputs (determinism)."""
        sp1 = compute_sp_total(grid_coords, grid_coords, layout_type="grid")
        sp2 = compute_sp_total(grid_coords, grid_coords, layout_type="grid")
        assert sp1 == sp2, "SP computation must be deterministic"
    
    def test_sp_random_destruction(self, grid_coords):
        """Test: Random relayout → SP ≈ 0."""
        rng = np.random.default_rng(42)
        coords_rand = rng.uniform(-1, 1, grid_coords.shape)
        sp = compute_sp_total(grid_coords, coords_rand, layout_type="grid")
        assert sp < 0.3, "Complete randomization should destroy structure"
    
    def test_sp_components_range(self, grid_coords):
        """Test: All SP components in [0, 1]."""
        comps = compute_sp_components(grid_coords, grid_coords, layout_type="grid")
        assert 0.0 <= comps.sp_adj <= 1.0
        assert 0.0 <= comps.sp_ord <= 1.0
        assert 0.0 <= comps.sp_clu <= 1.0
        assert 0.0 <= comps.total <= 1.0
    
    def test_knn_graph_symmetry(self, grid_coords):
        """Test: k-NN graph is symmetric."""
        adj = compute_knn_graph(grid_coords, k=4)
        assert np.allclose(adj, adj.T), "Adjacency matrix must be symmetric"
    
    def test_jaccard_identity(self):
        """Test: Jaccard(A, A) = 1.0."""
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
        jaccard = jaccard_similarity_bool(adj, adj)
        assert jaccard == pytest.approx(1.0), "Jaccard similarity of identical graphs = 1"
    
    def test_jaccard_disjoint(self):
        """Test: Jaccard(A, B) = 0 for disjoint graphs."""
        adj_a = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
        adj_b = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=bool)
        jaccard = jaccard_similarity_bool(adj_a, adj_b)
        assert jaccard == pytest.approx(0.0, abs=1e-6), "Disjoint graphs → Jaccard = 0"
    
    def test_sp_rotation_invariance(self, grid_coords):
        """Test: Rotation preserves SP (isometry)."""
        theta = np.pi / 4
        c, s = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([[c, -s], [s, c]])
        coords_rot = grid_coords @ rot_matrix.T
        
        sp = compute_sp_total(grid_coords, coords_rot, layout_type="grid")
        assert sp > 0.95, "Rotation should preserve structure"
    
    def test_sp_multiple_layouts(self):
        """Test: SP works across different layout types."""
        rng = np.random.default_rng(42)
        layouts = {
            "grid": np.random.uniform(-1, 1, (64, 2)),
            "line": np.random.uniform(-1, 1, (64, 2)),
            "circle": np.random.uniform(-1, 1, (64, 2)),
            "random": np.random.uniform(-1, 1, (64, 2)),
        }
        
        for layout_type, coords in layouts.items():
            sp = compute_sp_total(coords, coords, layout_type=layout_type)
            assert 0.0 <= sp <= 1.0, f"SP out of range for {layout_type}"


class TestSPEdgeCases:
    """Test edge cases and error handling."""
    
    def test_sp_single_point(self):
        """Test: Single point raises error."""
        coords = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError):
            compute_knn_graph(coords, k=1)
    
    def test_sp_mismatched_shapes(self):
        """Test: Mismatched shapes raise error."""
        coords_a = np.random.uniform(-1, 1, (64, 2))
        coords_b = np.random.uniform(-1, 1, (32, 2))
        
        with pytest.raises(AssertionError):
            compute_sp_total(coords_a, coords_b, layout_type="grid")
    
    def test_knn_k_too_large(self):
        """Test: k >= N raises error."""
        coords = np.random.uniform(-1, 1, (10, 2))
        with pytest.raises(ValueError):
            compute_knn_graph(coords, k=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
