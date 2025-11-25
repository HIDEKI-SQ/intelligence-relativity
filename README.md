# Intelligence Relativity

**Empirical Validation of the Relativity Theory of Intelligence**

[![Tests](https://github.com/HIDEKI-SQ/intelligence-relativity/actions/workflows/tests.yml/badge.svg)](https://github.com/HIDEKI-SQ/intelligence-relativity/actions/workflows/tests.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31019/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository contains the complete empirical foundation for **Intelligence Relativity**, a theoretical framework proposing that:

> **Structure (S)**, **Value (V)**, and **Meaning (M)** form a relativistic trinity where meaning emerges as the projection of structure under value illumination.

The framework establishes four fundamental observations:

- **O-1**: Natural Orthogonality (SSC ≈ 0 at λ=0)
- **O-2**: Topological Dominance (Phase > Metric)
- **O-3**: Stress Tolerance (Independent axes)
- **O-4**: Value-Gated Coupling (λ controls SSC)

### Two Measurement Systems

This repository implements two complementary measurement systems:

| System | Version | Metrics | Papers | Code |
|--------|---------|---------|--------|------|
| **SSC** | v1.x | Semantic-Spatial Correlation | I-1, O-1 | `experiments/` |
| **SP** | v2.x | Structural Preservation | I-2, O-2, O-3, O-4 | `experiments_sp/` |

**Both systems use Python 3.10.19 with identical core dependencies.**

---

## Repository Structure
```
intelligence-relativity/
├── src/
│   ├── core/                      # SSC measurement system (v1)
│   │   ├── ssc_computation.py     # SSC measurement
│   │   ├── deterministic.py       # Reproducibility
│   │   ├── generators.py          # Data generation
│   │   └── statistics.py          # Statistical tools
│   │
│   ├── core_sp/                   # SP measurement system (v2)
│   │   ├── sp_metrics.py          # SP computation
│   │   ├── ssc_wrapper.py         # SSC wrapper
│   │   ├── value_gate.py          # Value-gated coupling
│   │   ├── generators.py          # Embedding generation
│   │   ├── topology_ops.py        # Topological operations
│   │   ├── metric_ops.py          # Metric transformations
│   │   └── deterministic.py       # Environment verification
│   │
│   ├── experiments/               # SSC experiments (v1)
│   │   ├── exp_00_baseline.py             # O-1: Baseline
│   │   ├── exp_13_value_gate_sweep.py     # O-4: λ-sweep
│   │   ├── sup_exp_14_bert.py             # BERT validation
│   │   └── ... (17 experiments total)
│   │
│   └── experiments_sp/            # SP experiments (v2)
│       ├── i2_sp_instrument/              # I-2: Measurement system
│       │   ├── sp00_identity_isometry.py
│       │   ├── sp01_full_destruction.py
│       │   ├── sp02_topology_rewire_curve.py
│       │   └── sp03_layout_robustness.py
│       ├── o2_topological_dominance_sp/   # O-2: Topology
│       │   └── ... (4 experiments)
│       ├── o3_stress_independence_sp_ssc/ # O-3: Independence
│       │   └── ... (3 experiments)
│       ├── o4_value_gate_tradeoff_sp/     # O-4: Value gate (random layout)
│       │   ├── sp30_lambda_sweep_synth.py
│       │   └── sp31_lambda_sweep_bert.py
│       ├── o4_extra_sp/                   # O-4 Extra: Grid layout ✨ NEW
│       │   ├── sp50_lambda_tradeoff_grid_synth.py
│       │   └── sp51_lambda_tradeoff_grid_bert.py
│       ├── sp_robustness/                 # Robustness
│       │   └── ... (3 experiments)
│       └── generate_summary_all.py        # Summary generation
│
├── demos/                         # Application demos
│   └── dr_evaluation/             # Dimensionality reduction evaluation
│       ├── run_dr_demo.py         # Main demo script
│       ├── load_data.py           # MNIST data loading
│       ├── apply_dr.py            # t-SNE/UMAP/PCA
│       ├── compute_metrics.py     # SP/SSC computation
│       └── visualize_results.py   # Figure generation
│
├── tests/                         # Test suite (175 tests)
│   ├── test_core.py              # SSC system tests
│   ├── test_core_sp/             # SP system tests
│   ├── test_experiments_sp/      # SP experiment tests
│   └── test_integration/         # Integration tests
│
├── outputs/                       # v1 experiment results
├── outputs_sp/                    # v2 experiment results
│   ├── summary_all_I2.csv
│   ├── summary_all_O2.csv
│   ├── summary_all_O3.csv
│   ├── summary_all_O4.csv         # Random layout
│   ├── summary_all_O4_extra.csv   # Grid layout ✨ NEW
│   ├── summary_all_robust.csv
│   └── env.txt                    # Environment record
│
├── .github/workflows/
│   ├── tests.yml                 # CI testing
│   ├── run_experiments.yml       # v1 experiments
│   ├── run_experiments_sp.yml    # v2 experiments
│   └── test_demo.yml             # Demo testing
│
├── requirements.txt              # Unified dependencies
├── CHANGELOG.md                  # Version history
└── README.md                     # This file
```

---

## Installation

### Requirements

- Python 3.10.19
- NumPy 1.24.3
- SciPy 1.10.1
- (Optional) Transformers + PyTorch for BERT experiments
- (Optional) umap-learn for dimensionality reduction demos

### Setup
```bash
# Clone repository
git clone https://github.com/HIDEKI-SQ/intelligence-relativity.git
cd intelligence-relativity

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### SSC Experiments (v1)
```bash
# Run baseline experiment (O-1)
python -m src.experiments.exp_00_baseline
```

**Output:**
```
SSC: -0.0025 ± 0.0736
90% CI: [-0.0063, 0.0013]
✅ Natural orthogonality confirmed
```

### SP Experiments (v2)
```bash
# Run identity/isometry baseline (I-2)
python -m src.experiments_sp.i2_sp_instrument.sp00_identity_isometry
```

**Output:**
```
SP_total: 0.879 ± 0.014
✅ High structural preservation confirmed
```

### O-4 Extra: Grid Layout Experiments (v2.1.0)
```bash
# Run grid layout lambda trade-off (synthetic)
python -m src.experiments_sp.o4_extra_sp.sp50_lambda_tradeoff_grid_synth

# Run grid layout lambda trade-off (BERT)
python -m src.experiments_sp.o4_extra_sp.sp51_lambda_tradeoff_grid_bert
```

**Output:**
```
Lambda = 0.0
  SP:  0.8500 ± 0.0020
  SSC: 0.0001 ± 0.0002

Lambda = 1.0
  SP:  0.6000 ± 0.0030
  SSC: 0.1500 ± 0.0040
```

### Dimensionality Reduction Demo
```bash
# Compare t-SNE, UMAP, PCA on MNIST
python demos/dr_evaluation/run_dr_demo.py
```

**Output:**
- `demos/outputs/dr_evaluation/results.csv`
- `demos/outputs/dr_evaluation/comparison_bar.png`
- `demos/outputs/dr_evaluation/embeddings_scatter.png`

### Run Tests
```bash
# Run all tests (175 tests)
pytest tests/ -v

# Run specific subsystem
pytest tests/test_core.py -v              # SSC system
pytest tests/test_core_sp/ -v             # SP system
pytest tests/test_experiments_sp/ -v      # SP experiments
```

---

## Core Concepts

### Semantic-Spatial Correlation (SSC)

Primary measurement for O-1 validation:
```python
from src.core import compute_ssc

# Compute SSC between semantic and spatial distances
ssc = compute_ssc(semantic_distances, spatial_distances)

# Expected: SSC ≈ 0 at λ=0 (Natural Orthogonality)
```

### Structural Preservation (SP)

Three-component measurement for structure preservation:
```python
from src.core_sp import compute_sp_total

# Compute SP between baseline and transformed coordinates
sp = compute_sp_total(
    base_coords=original_coords,
    trans_coords=transformed_coords,
    layout_type="cluster"
)

# Components: SP_adj (adjacency), SP_ord (order), SP_clu (clustering)
# Returns: SP_total in [0, 1]
```

### Value-Gated Coupling

Control semantic-spatial coupling via λ parameter:
```python
from src.core_sp import apply_value_gate

# Apply value gate to coordinates
coords_gated = apply_value_gate(
    base_coords=coords,
    embeddings=semantic_embeddings,
    lam=0.5  # 0=no coupling, 1=full coupling
)

# Expected: SSC increases monotonically with λ (O-4)
```

---

## Demos

### Dimensionality Reduction Evaluation

Demonstrates SP and SSC metrics on real data (MNIST):

**What it does:**
- Loads MNIST subset (configurable size)
- Applies t-SNE, UMAP, PCA
- Computes SP and SSC for each method
- Generates comparison figures

**Example results:**

| Method | SP_total | SSC | Interpretation |
|--------|----------|-----|----------------|
| t-SNE  | 0.43     | 0.38 | Strong local structure |
| UMAP   | 0.27     | 0.35 | Order preservation |
| PCA    | 0.23     | 0.49 | Global distances only |

**Key finding:** SP and SSC measure independent dimensions—PCA shows lowest SP but highest SSC, confirming that structural preservation and distance correlation are distinct metrics.

---

## Reproducibility

### Deterministic Execution

All experiments execute with:
- **Fixed seeds**: Every random operation reproducible
- **Locked dependencies**: Exact versions in `requirements.txt`
- **Single-threaded BLAS**: Eliminates non-determinism
- **Environment logging**: Automatic `env.txt` generation

### Example
```python
from src.core_sp import set_deterministic_mode, verify_environment

# Enable deterministic mode
set_deterministic_mode()

# Generate environment record
verify_environment("outputs_sp/env.txt")
```

**Output (`env.txt`):**
```json
{
  "python": "3.10.19",
  "numpy": "1.24.3",
  "scipy": "1.10.1",
  "blas_threads": {"OPENBLAS_NUM_THREADS": "1"}
}
```

### CI/CD

- **Tests**: Run on every push (175 tests)
- **Experiments**: On-demand via GitHub Actions
- **Standard deviation**: 0.00 across runs

---

## Experiment Results

### v1 (SSC System)

Complete results in `outputs/`:
- 17 experiments validated
- O-1 confirmed across layouts/metrics/seeds
- BERT and multilingual validation

### v2 (SP System)

Complete results in `outputs_sp/`:
- 18 experiments (I-2, O-2, O-3, O-4, O-4 Extra, Robustness)
- Summary files: `summary_all_*.csv`
- Environment: `env.txt`

**O-4 Validation:** Two complementary conditions
- **Random layout** (sp30-sp31): Control for initial structure absence
- **Grid layout** (sp50-sp51): Demonstrates trade-off from structured baseline

---

## Citation
```bibtex
@software{intelligence_relativity_2025,
  author = {Hideki},
  title = {Intelligence Relativity: Empirical Validation},
  version = {2.1.0},
  year = {2025},
  url = {https://github.com/HIDEKI-SQ/intelligence-relativity},
  note = {Complete validation framework for SSC and SP measurement systems}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contributing

This repository contains validated measurement systems. For extensions:

1. Fork the repository
2. Run test suite: `pytest tests/ -v`
3. Add your extension
4. Ensure deterministic reproducibility
5. Submit pull request

---

## Contact

**Author**: Hideki  
**Email**: hideki@r3776.jp  
**GitHub**: [@HIDEKI-SQ](https://github.com/HIDEKI-SQ)  
**Repository**: https://github.com/HIDEKI-SQ/intelligence-relativity  

For questions or bug reports, please open an [issue](https://github.com/HIDEKI-SQ/intelligence-relativity/issues).  
For collaboration inquiries, feel free to contact via email.

**Framework**: Relativity Theory of Intelligence

---

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

**Latest:** v2.1.0 - O-4 Extra experiments (grid layout validation)

---

**Complete empirical foundation for Intelligence Relativity**

*Deterministic reproducibility across all experiments (Python 3.10.19, NumPy 1.24.3)*
