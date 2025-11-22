"""Load MNIST subset for dimensionality reduction demo.

Uses scikit-learn's fetch_openml to automatically download MNIST.
Data is cached locally after first download.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from typing import Tuple


def load_mnist_subset(
    n_per_class: int = 50,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load balanced MNIST subset.
    
    Parameters
    ----------
    n_per_class : int, default=50
        Number of samples per class (0-9)
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    X : ndarray, shape (n_per_class * 10, 784)
        Flattened pixel values (0-255)
    y : ndarray, shape (n_per_class * 10,)
        Class labels (0-9)
    
    Examples
    --------
    >>> X, y = load_mnist_subset(n_per_class=50)
    >>> X.shape
    (500, 784)
    >>> y.shape
    (500,)
    """
    print(f"📊 Loading MNIST subset ({n_per_class} samples per class)...")
    
    # Fetch MNIST (cached after first download)
    mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
    X_full = mnist.data
    y_full = mnist.target.astype(int)
    
    # Sample balanced subset
    rng = np.random.default_rng(random_state)
    indices = []
    
    for digit in range(10):
        digit_indices = np.where(y_full == digit)[0]
        selected = rng.choice(digit_indices, size=n_per_class, replace=False)
        indices.extend(selected)
    
    indices = np.array(indices)
    X = X_full[indices]
    y = y_full[indices]
    
    print(f"  ✅ Loaded: {X.shape}")
    print(f"  Classes: {np.unique(y)}")
    
    return X, y


if __name__ == "__main__":
    # Test
    X, y = load_mnist_subset(n_per_class=10)
    print(f"\nTest successful!")
    print(f"X range: [{X.min():.1f}, {X.max():.1f}]")
    print(f"Class distribution: {np.bincount(y)}")
