"""Core SP (Structural Preservation) measurement module.

This module provides SP measurement instruments complementary to
SSC (Semantic-Spatial Correlation) in the v1 core module.

Modules:
    sp_metrics: SP computation (adjacency/order/cluster)
    topology_ops: Topology disruption operations
    metric_ops: Metric transformation operations
    ssc_wrapper: SSC computation wrapper (v1 compatibility)
    value_gate: Value gate mechanism wrapper
    generators: Semantic data generation utilities
    deterministic: Deterministic execution and environment verification
"""

from .sp_metrics import compute_sp_total
from .topology_ops import edge_rewire, permute_coords, random_relayout
from .metric_ops import rotate_2d, scale_2d, shear_2d, add_coord_noise
from .ssc_wrapper import compute_ssc_wrapper
from .value_gate import apply_value_gate
from .generators import generate_semantic_embeddings, add_semantic_noise
from .deterministic import set_deterministic_mode, verify_environment, generate_manifest

__version__ = "2.0.0"

__all__ = [
    "compute_sp_total",
    "edge_rewire",
    "permute_coords",
    "random_relayout",
    "rotate_2d",
    "scale_2d",
    "shear_2d",
    "add_coord_noise",
    "compute_ssc_wrapper",
    "apply_value_gate",
    "generate_semantic_embeddings",
    "add_semantic_noise",
    "set_deterministic_mode",
    "verify_environment",
    "generate_manifest",
]
