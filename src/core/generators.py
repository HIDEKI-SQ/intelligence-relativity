"""Standard data generators.

All generators use numpy.random.Generator with explicit seeds.
No global random state is used.

See README.md for layout specifications.
"""

import numpy as np


def generate_embeddings(
    n_items: int,
    dim: int,
    seed: int,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate random semantic embeddings.
    
    Parameters
    ----------
    n_items : int
        Number of items
    dim : int
        Embedding dimension
    seed : int
        Random seed for reproducibility
    normalize : bool, default=True
        If True, normalize each row to unit length
    
    Returns
    -------
    embeddings : ndarray, shape (n_items, dim)
        Random embeddings from standard normal distribution
    
    Examples
    --------
    >>> embeddings = generate_embeddings(64, 128, seed=42)
    >>> embeddings.shape
    (64, 128)
    """
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n_items, dim))
    
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-12)
    
    return embeddings


def generate_spatial_coords(
    n_items: int,
    layout: str,
    seed: int,
    radius: float = 1.0
) -> np.ndarray:
    """
    Generate spatial coordinates with specified layout.
    
    Parameters
    ----------
    n_items : int
        Number of items
    layout : str
        'random', 'circle', 'grid', 'line', or 'cube'
    seed : int
        Random seed for reproducibility
    radius : float, default=1.0
        Layout scale parameter
    
    Returns
    -------
    coords : ndarray
        Spatial coordinates, shape (n_items, 2) or (n_items, 3)
    
    Examples
    --------
    >>> coords = generate_spatial_coords(64, 'circle', seed=42)
    >>> coords.shape
    (64, 2)
    """
    rng = np.random.default_rng(seed)
    
    if layout == 'random':
        return rng.uniform(-radius, radius, (n_items, 2))
    
    elif layout == 'circle':
        ordering = rng.permutation(n_items)
        angles = 2 * np.pi * np.arange(n_items) / n_items
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        coords = np.column_stack([x, y])
        ordered_coords = np.zeros_like(coords)
        ordered_coords[ordering] = coords
        return ordered_coords
    
    elif layout == 'grid':
        grid_size = int(np.ceil(np.sqrt(n_items)))
        x_pos = np.arange(grid_size)
        y_pos = np.arange(grid_size)
        xx, yy = np.meshgrid(x_pos, y_pos)
        grid_positions = np.column_stack([xx.ravel(), yy.ravel()])[:n_items]
        rng.shuffle(grid_positions)
        if grid_size > 1:
            grid_positions = 2 * radius * grid_positions / (grid_size - 1) - radius
        return grid_positions.astype(np.float64)
    
    elif layout == 'line':
        ordering = rng.permutation(n_items)
        positions = np.linspace(0, radius, n_items)
        coords = np.column_stack([positions, np.zeros(n_items)])
        ordered_coords = np.zeros_like(coords)
        ordered_coords[ordering] = coords
        return ordered_coords
    
    elif layout == 'cube':
        grid_size = int(np.ceil(n_items ** (1/3)))
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(positions) < n_items:
                        positions.append([i, j, k])
        coords = np.array(positions[:n_items], dtype=float)
        if grid_size > 1:
            coords = radius * coords / (grid_size - 1)
        rng.shuffle(coords)
        return coords
    
    else:
        raise ValueError(f"Unknown layout: {layout}")
