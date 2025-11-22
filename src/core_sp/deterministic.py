"""Deterministic execution utilities.

Ensures bit-level reproducibility using numpy.random.Generator.
No global random state is used.

Validated in I-1. See README.md for details.
"""

import os
import sys
import platform
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json


def set_deterministic_mode(verbose: bool = True) -> None:
    """
    Set deterministic execution mode.
    
    Configures environment for single-threaded BLAS execution.
    
    Parameters
    ----------
    verbose : bool, default=True
        If True, print configuration
    
    Notes
    -----
    Random number generation uses numpy.random.Generator with
    explicit seeds in each function, not global state.
    
    Examples
    --------
    >>> set_deterministic_mode()
    >>> embeddings = generate_embeddings(64, 128, seed=42)
    """
    # Single-threaded BLAS
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    if verbose:
        print("✅ Deterministic mode: single-threaded BLAS")
        print("   Use seed parameter in each function for reproducibility")


def verify_environment(output_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Verify and record execution environment.
    
    Parameters
    ----------
    output_path : Path, optional
        If provided, save environment info as JSON
    
    Returns
    -------
    env_info : dict
        Environment information including BLAS configuration
    
    Examples
    --------
    >>> env = verify_environment(Path("outputs/env.txt"))
    """
    import scipy
    
    # Basic versions
    env_info = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
    }
    
    # Optional: pandas
    try:
        import pandas as pd
        env_info['pandas'] = pd.__version__
    except ImportError:
        pass
    
    # BLAS configuration
    env_info['blas_threads'] = {
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS', 'not set'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS', 'not set'),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', 'not set'),
    }
    
    # BLAS library info
    try:
        import numpy.distutils.system_info as sysinfo
        blas_info = sysinfo.get_info('blas_opt')
        if blas_info:
            env_info['blas_library'] = blas_info.get('name', 'unknown')
    except:
        env_info['blas_library'] = 'unable to detect'
    
    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(env_info, f, indent=2)
        print(f"✅ Environment saved: {output_path}")
    
    return env_info


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Parameters
    ----------
    file_path : Path
        Path to file
    
    Returns
    -------
    hash_str : str
        SHA-256 hash as hexadecimal string
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_manifest(
    output_dir: Path,
    manifest_path: Optional[Path] = None
) -> Dict[str, str]:
    """
    Generate SHA-256 manifest for output directory.
    
    Parameters
    ----------
    output_dir : Path
        Directory containing output files
    manifest_path : Path, optional
        If provided, save manifest as JSON
    
    Returns
    -------
    manifest : dict
        Mapping from filename to SHA-256 hash
    
    Examples
    --------
    >>> manifest = generate_manifest(
    ...     Path("outputs/exp00"),
    ...     Path("outputs/exp00/manifest.json")
    ... )
    """
    manifest = {}
    
    for file_path in sorted(output_dir.glob("*")):
        if file_path.is_file() and file_path.suffix != '.json':
            rel_path = file_path.relative_to(output_dir)
            manifest[str(rel_path)] = compute_file_hash(file_path)
    
    if manifest_path is not None:
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"✅ Manifest saved: {manifest_path}")
    
    return manifest
