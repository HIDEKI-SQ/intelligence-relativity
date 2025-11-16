# Intelligence Relativity

**Empirical Validation of the Relativity Theory of Intelligence**

[![Tests](https://github.com/HIDEKI-SQ/intelligence-relativity/actions/workflows/tests.yml/badge.svg)](https://github.com/HIDEKI-SQ/intelligence-relativity/actions/workflows/tests.yml)
[![Experiments](https://github.com/HIDEKI-SQ/intelligence-relativity/actions/workflows/run_experiments.yml/badge.svg)](https://github.com/HIDEKI-SQ/intelligence-relativity/actions/workflows/run_experiments.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
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

---

## Key Results

### ✅ 17/17 Experiments Completed
```
Total Experiments: 17
Success Rate: 100.0%
Deterministic Reproducibility: Confirmed
```

### Core Findings

| Observation | Description | Status |
|-------------|-------------|--------|
| **O-1** | Natural Orthogonality (SSC ≈ 0 at λ=0) | ✅ Validated (11 experiments) |
| **O-2** | Topological Dominance (Phase > Metric) | ✅ Validated (2 experiments) |
| **O-3** | Stress Tolerance (Independent axes) | ✅ Validated (2 experiments) |
| **O-4** | Value-Gated Coupling (λ controls SSC) | ✅ Validated (2 experiments) |

### Real-World Validation

- **BERT Embeddings** (SUP-14): O-1 confirmed with SSC = 0.0031 ± 0.0614
- **Cross-Linguistic** (SUP-15): English, Japanese, Chinese validation
- **λ-Sweep** (EXP-13, SUP-14): Monotonic coupling control demonstrated

---

## Repository Structure
```
intelligence-relativity/
├── src/
│   ├── core/                      # Core measurement toolkit
│   │   ├── __init__.py
│   │   ├── ssc_computation.py     # SSC measurement (validated in I-1)
│   │   ├── deterministic.py       # Reproducibility guarantees
│   │   ├── generators.py          # Data generation utilities
│   │   ├── statistics.py          # Statistical tools
│   │   └── visualization.py       # Plotting utilities
│   │
│   └── experiments/               # Complete experiment suite
│       ├── exp_00_baseline.py             # O-1: Baseline
│       ├── exp_01_spatial_vs_random.py    # O-1: Robustness
│       ├── exp_02_grid_arrangement.py     # O-1: Grid
│       ├── exp_03_line_arrangement.py     # O-1: Line
│       ├── exp_04_3d_cube_arrangement.py  # O-1: 3D
│       ├── exp_05_independence_permutation.py  # O-1: Permutation
│       ├── exp_06_dimension_robustness.py # O-1: Dimensions
│       ├── exp_07_sample_size_robustness.py   # O-1: Sample size
│       ├── exp_08_metric_robustness.py    # O-1: Metrics
│       ├── exp_09_topological_disruption.py   # O-2: Topology
│       ├── exp_10_rotation_invariance.py  # O-2: Invariance
│       ├── exp_11_coordinate_noise.py     # O-3: Coord noise
│       ├── exp_12_semantic_noise.py       # O-3: Semantic noise
│       ├── exp_13_value_gate_sweep.py     # O-4: λ-sweep
│       ├── sup_exp_14_bert.py             # SUP-14: BERT
│       ├── sup_exp_15_multilingual.py     # SUP-15: Multilingual
│       └── exp_beta_initial_exploration.py  # Beta: Historical
│
├── tests/                         # Test suite
│   ├── test_core.py              # Core toolkit tests
│   ├── test_ssc_computation.py   # SSC measurement tests
│   ├── test_deterministic.py     # Reproducibility tests
│   └── test_experiments.py       # Experiment execution tests
│
├── outputs/                       # Experiment results (generated)
│   ├── exp00_baseline/
│   ├── sup14_bert/
│   ├── sup15_multilingual/
│   ├── master_summary.json       # Complete results
│   └── EXPERIMENT_OVERVIEW.md    # Detailed overview
│
├── .github/workflows/
│   ├── tests.yml                 # CI/CD testing
│   └── run_experiments.yml       # Full experiment suite
│
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

---

## Installation

### Requirements

- Python 3.11+
- NumPy 2.3+
- SciPy 1.16+
- (Optional) Transformers + PyTorch for BERT experiments

### Setup
```bash
# Clone repository
git clone https://github.com/HIDEKI-SQ/intelligence-relativity.git
cd intelligence-relativity

# Install dependencies
pip install -r requirements.txt

# Optional: Install BERT dependencies
pip install transformers torch
```

---

## Quick Start

### Run Single Experiment
```bash
# Run baseline experiment (EXP-00)
python -m src.experiments.exp_00_baseline
```

**Output**:
```
EXP-00: Baseline (Natural Orthogonality)
SSC: -0.0025 ± 0.0736
90% CI: [-0.0063, 0.0013]
✅ Natural orthogonality confirmed
```

### Run Complete Suite
```bash
# Run all 17 experiments via GitHub Actions
# or manually:
for exp in src/experiments/exp_*.py; do
    python -m ${exp%.py}
done
```

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_core.py -v
pytest tests/test_ssc_computation.py -v
```

---

## Core Concepts

### Semantic-Spatial Correlation (SSC)

The primary measurement instrument validated in I-1:
```python
from src.core import compute_ssc
from scipy.spatial.distance import pdist

# Compute SSC
semantic_distances = pdist(embeddings, 'correlation')
spatial_distances = pdist(coordinates, 'euclidean')

ssc = compute_ssc(semantic_distances, spatial_distances)
# Expected: SSC ≈ 0 at λ=0 (Natural Orthogonality)
```

### Observation Framework
```python
# O-1: Natural Orthogonality
# At λ=0, semantic and spatial structures are independent
assert abs(ssc) < 0.10  # Quasi-orthogonal domain

# O-4: Value-Gated Coupling
# λ monotonically increases SSC
lambda_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Expected: SSC increases with λ
```

---

## Reproducibility

### Deterministic Execution

All experiments execute with:
- **Fixed seeds**: Every random operation uses specified seed
- **Locked dependencies**: `requirements.txt` pins exact versions
- **Single-threaded BLAS**: Eliminates non-determinism
- **Standard deviation = 0.00**: Perfect reproducibility

### Verification
```python
from src.core import set_deterministic_mode, verify_environment

# Enable deterministic mode
set_deterministic_mode()

# Verify environment
verify_environment("outputs/env.txt")
```

### CI/CD

- **Tests**: Automated on every push
- **Experiments**: On-demand execution via GitHub Actions
- **Results**: Archived with SHA256 manifests

---

## Experiment Results

### Master Summary

Complete results available in:
- **JSON**: `outputs/master_summary.json`
- **Markdown**: `outputs/EXPERIMENT_OVERVIEW.md`

### Key Statistics

| Category | Experiments | Success Rate |
|----------|-------------|--------------|
| Core (EXP-00 to EXP-13) | 14 | 100% |
| Supplementary (SUP-14, SUP-15) | 2 | 100% |
| Beta (Historical) | 1 | 100% |

### Sample Results

**EXP-00 (Baseline)**:
```json
{
  "mean": -0.0025,
  "std": 0.0736,
  "ci_90": [-0.0063, 0.0013],
  "n": 1000
}
```

**SUP-14 (BERT, O-4 λ-sweep)**:
```json
{
  "lambda": 1.0,
  "mean": 0.1345,
  "ci_90": [0.1336, 0.1354]
}
```

---

## Citation

If you use this work, please cite:
```bibtex
@software{intelligence_relativity_2025,
  author = {Hideki},
  title = {Intelligence Relativity: Empirical Validation},
  year = {2025},
  url = {https://github.com/HIDEKI-SQ/intelligence-relativity},
  note = {Complete experimental validation of the Relativity Theory of Intelligence}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contributing

This repository represents a complete empirical foundation. For theoretical extensions or applications:

1. Fork the repository
2. Run the test suite: `pytest tests/ -v`
3. Add your extension
4. Submit a pull request

---

## Contact

**Author**: Hideki  
**Repository**: https://github.com/HIDEKI-SQ/intelligence-relativity  
**Framework**: Optics of Intelligence

---

## Acknowledgments

- **Observation Framework**: O-1 through O-4 validated across 17 experiments
- **Real-World Data**: BERT embeddings confirm theoretical predictions
- **Cross-Linguistic**: English, Japanese, Chinese universality demonstrated
- **Reproducibility**: 100% deterministic execution with CI/CD verification

---

**This repository represents the complete empirical foundation for Intelligence Relativity.**

*All experiments executed with deterministic reproducibility (fixed seeds, locked dependencies)*
