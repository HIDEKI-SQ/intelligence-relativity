# src/core_sp/metric_ops.py

from __future__ import annotations

import numpy as np
from typing import Tuple


def rotate_2d(
    coords: np.ndarray,
    theta_rad: float,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Rotate 2D coordinates around a center by theta (radians).

    Parameters
    ----------
    coords : (N, 2)
    theta_rad : float
    center : (cx, cy)

    Returns
    -------
    coords_new : (N, 2)
    """
    coords = np.asarray(coords, dtype=float)
    assert coords.shape[1] == 2
    cx, cy = center
    x = coords[:, 0] - cx
    y = coords[:, 1] - cy
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    x_new = c * x - s * y
    y_new = s * x + c * y
    return np.stack([x_new + cx, y_new + cy], axis=1)


def scale_2d(
    coords: np.ndarray,
    sx: float,
    sy: float,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Anisotropic scaling around a center.

    Parameters
    ----------
    coords : (N, 2)
    sx, sy : float
    center : (cx, cy)

    Returns
    -------
    coords_new : (N, 2)
    """
    coords = np.asarray(coords, dtype=float)
    assert coords.shape[1] == 2
    cx, cy = center
    x = coords[:, 0] - cx
    y = coords[:, 1] - cy
    x_new = sx * x
    y_new = sy * y
    return np.stack([x_new + cx, y_new + cy], axis=1)


def shear_2d(
    coords: np.ndarray,
    k: float,
) -> np.ndarray:
    """
    Simple shear transform: x' = x + k*y, y' = y.

    Parameters
    ----------
    coords : (N, 2)
    k : float

    Returns
    -------
    coords_new : (N, 2)
    """
    coords = np.asarray(coords, dtype=float)
    assert coords.shape[1] == 2
    x = coords[:, 0]
    y = coords[:, 1]
    x_new = x + k * y
    return np.stack([x_new, y], axis=1)


def add_coord_noise(
    coords: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
) -> np.ndarray:
    """
    Add isotropic Gaussian noise to coordinates.

    Parameters
    ----------
    coords : (N, d)
    rng : np.random.Generator
    sigma : float

    Returns
    -------
    coords_new : (N, d)
    """
    coords = np.asarray(coords, dtype=float)
    noise = rng.normal(loc=0.0, scale=sigma, size=coords.shape)
    return coords + noise
