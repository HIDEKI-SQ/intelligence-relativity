"""Tests for metric operations.

Validates:
    - Rotation preserves distances
    - Scaling transforms correctly
    - Shear preserves topology
    - Coordinate noise determinism
"""

import pytest
import numpy as np
from src.core_sp.metric_ops import rotate_2d, scale_2d, shear_2d, add_coord_noise


class TestMetricOps:
    """Test suite for metric operations."""
    
    @pytest.fixture
    def grid_coords(self):
        """4x4 grid for faster tests."""
        xs = np.linspace(-1, 1, 4)
        ys = np.linspace(-1, 1, 4)
        xv, yv = np.meshgrid(xs, ys)
        return np.stack([xv.ravel(), yv.ravel()], axis=1)
    
    def test_rotate_preserves_distances(self, grid_coords):
        """Test: Rotation is isometric (preserves pairwise distances)."""
        from scipy.spatial.distance import pdist
        
        coords_rot = rotate_2d(grid_coords, theta_rad=np.pi / 4, center=(0, 0))
        
        d_orig = pdist(grid_coords)
        d_rot = pdist(coords_rot)
        
        assert np.allclose(d_orig, d_rot, atol=1e-10), "Rotation must preserve distances"
    
    def test_rotate_determinism(self, grid_coords):
        """Test: Same rotation → same result."""
        coords1 = rotate_2d(grid_coords, theta_rad=np.pi / 3, center=(0, 0))
        coords2 = rotate_2d(grid_coords, theta_rad=np.pi / 3, center=(0, 0))
        assert np.allclose(coords1, coords2), "Rotation must be deterministic"
    
    def test_rotate_shape(self, grid_coords):
        """Test: Rotation preserves shape."""
        coords_rot = rotate_2d(grid_coords, theta_rad=np.pi / 2, center=(0, 0))
        assert coords_rot.shape == grid_coords.shape
    
    def test_scale_uniform(self, grid_coords):
        """Test: Uniform scaling multiplies distances."""
        from scipy.spatial.distance import pdist
        
        coords_scaled = scale_2d(grid_coords, sx=2.0, sy=2.0, center=(0, 0))
        
        d_orig = pdist(grid_coords)
        d_scaled = pdist(coords_scaled)
        
        assert np.allclose(d_scaled, 2.0 * d_orig, atol=1e-10), "Uniform scaling scales distances"
    
    def test_scale_determinism(self, grid_coords):
        """Test: Same scaling → same result."""
        coords1 = scale_2d(grid_coords, sx=1.5, sy=2.0, center=(0, 0))
        coords2 = scale_2d(grid_coords, sx=1.5, sy=2.0, center=(0, 0))
        assert np.allclose(coords1, coords2), "Scaling must be deterministic"
    
    def test_shear_preserves_topology(self, grid_coords):
        """Test: Shear preserves neighbor relationships."""
        from src.core_sp.sp_metrics import compute_knn_graph
        
        coords_sheared = shear_2d(grid_coords, k=0.5)
        
        adj_orig = compute_knn_graph(grid_coords, k=3)
        adj_shear = compute_knn_graph(coords_sheared, k=3)
        
        # High overlap expected (not 100% due to boundary effects)
        overlap = (adj_orig & adj_shear).sum() / adj_orig.sum()
        assert overlap > 0.7, "Shear should preserve most neighbor relationships"
    
    def test_shear_determinism(self, grid_coords):
        """Test: Same shear → same result."""
        coords1 = shear_2d(grid_coords, k=0.7)
        coords2 = shear_2d(grid_coords, k=0.7)
        assert np.allclose(coords1, coords2), "Shear must be deterministic"
    
    def test_add_coord_noise_determinism(self, grid_coords):
        """Test: Same seed → same noise."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        coords1 = add_coord_noise(grid_coords, rng=rng1, sigma=0.1)
        coords2 = add_coord_noise(grid_coords, rng=rng2, sigma=0.1)
        
        assert np.allclose(coords1, coords2), "Noise must be deterministic"
    
    def test_add_coord_noise_sigma0(self, grid_coords):
        """Test: σ=0 → no change."""
        rng = np.random.default_rng(42)
        coords_noisy = add_coord_noise(grid_coords, rng=rng, sigma=0.0)
        assert np.allclose(coords_noisy, grid_coords), "σ=0 should add no noise"
    
    def test_add_coord_noise_shape(self, grid_coords):
        """Test: Noise preserves shape."""
        rng = np.random.default_rng(42)
        coords_noisy = add_coord_noise(grid_coords, rng=rng, sigma=0.1)
        assert coords_noisy.shape == grid_coords.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
