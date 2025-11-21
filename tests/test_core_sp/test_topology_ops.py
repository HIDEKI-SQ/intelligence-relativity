"""Tests for topology operations.

Validates:
    - Edge rewiring preserves symmetry
    - Permutation determinism
    - Random relayout bounds
"""

import pytest
import numpy as np
from src.core_sp.topology_ops import edge_rewire, permute_coords, random_relayout


class TestTopologyOps:
    """Test suite for topology operations."""
    
    @pytest.fixture
    def adjacency_matrix(self):
        """Sample symmetric adjacency matrix."""
        adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ], dtype=bool)
        return adj
    
    @pytest.fixture
    def grid_coords(self):
        """8x8 grid."""
        xs = np.linspace(-1, 1, 8)
        ys = np.linspace(-1, 1, 8)
        xv, yv = np.meshgrid(xs, ys)
        return np.stack([xv.ravel(), yv.ravel()], axis=1)
    
    def test_edge_rewire_symmetry(self, adjacency_matrix):
        """Test: Rewired graph remains symmetric."""
        rng = np.random.default_rng(42)
        adj_new = edge_rewire(adjacency_matrix, p=0.5, rng=rng)
        assert np.allclose(adj_new, adj_new.T), "Rewired graph must be symmetric"
    
    def test_edge_rewire_p0(self, adjacency_matrix):
        """Test: p=0 → no change."""
        rng = np.random.default_rng(42)
        adj_new = edge_rewire(adjacency_matrix, p=0.0, rng=rng)
        assert np.array_equal(adj_new, adjacency_matrix), "p=0 should preserve graph"
    
    def test_edge_rewire_determinism(self, adjacency_matrix):
        """Test: Same seed → same rewiring."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        adj1 = edge_rewire(adjacency_matrix, p=0.5, rng=rng1)
        adj2 = edge_rewire(adjacency_matrix, p=0.5, rng=rng2)
        assert np.array_equal(adj1, adj2), "Rewiring must be deterministic"
    
    def test_permute_coords_determinism(self, grid_coords):
        """Test: Same seed → same permutation."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        coords1 = permute_coords(grid_coords, rng=rng1, p=0.5)
        coords2 = permute_coords(grid_coords, rng=rng2, p=0.5)
        assert np.allclose(coords1, coords2), "Permutation must be deterministic"
    
    def test_permute_coords_p0(self, grid_coords):
        """Test: p=0 → no change."""
        rng = np.random.default_rng(42)
        coords_new = permute_coords(grid_coords, rng=rng, p=0.0)
        assert np.allclose(coords_new, grid_coords), "p=0 should preserve coords"
    
    def test_permute_coords_shape(self, grid_coords):
        """Test: Permutation preserves shape."""
        rng = np.random.default_rng(42)
        coords_new = permute_coords(grid_coords, rng=rng, p=0.5)
        assert coords_new.shape == grid_coords.shape, "Shape must be preserved"
    
    def test_random_relayout_bounds(self, grid_coords):
        """Test: Random relayout respects bounds."""
        rng = np.random.default_rng(42)
        coords_new = random_relayout(grid_coords, rng=rng, bounds=(-2, 2))
        assert np.all(coords_new >= -2) and np.all(coords_new <= 2), "Bounds violated"
    
    def test_random_relayout_determinism(self, grid_coords):
        """Test: Same seed → same relayout."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        coords1 = random_relayout(grid_coords, rng=rng1, bounds=(-1, 1))
        coords2 = random_relayout(grid_coords, rng=rng2, bounds=(-1, 1))
        assert np.allclose(coords1, coords2), "Relayout must be deterministic"
    
    def test_random_relayout_shape(self, grid_coords):
        """Test: Relayout preserves shape."""
        rng = np.random.default_rng(42)
        coords_new = random_relayout(grid_coords, rng=rng, bounds=(-1, 1))
        assert coords_new.shape == grid_coords.shape, "Shape must be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
