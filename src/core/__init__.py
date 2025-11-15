"""Core measurement toolkit for Optics of Intelligence.

See README.md for design philosophy and usage examples.
"""

from .ssc_computation import compute_ssc, compute_ssc_from_data
from .deterministic import set_deterministic_mode, verify_environment, generate_manifest
from .generators import generate_embeddings, generate_spatial_coords
from .statistics import bootstrap_ci, tost_equivalence, compute_summary_stats
from .visualization import plot_histogram, plot_scatter, plot_ci

__version__ = "1.0.0"
__all__ = [
    "compute_ssc",
    "compute_ssc_from_data",
    "set_deterministic_mode",
    "verify_environment",
    "generate_manifest",
    "generate_embeddings",
    "generate_spatial_coords",
    "bootstrap_ci",
    "tost_equivalence",
    "compute_summary_stats",
    "plot_histogram",
    "plot_scatter",
    "plot_ci",
]
