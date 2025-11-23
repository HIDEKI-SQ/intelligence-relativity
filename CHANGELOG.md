# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2025-11-22

### Added

**Major Release: SP (Structural Preservation) Measurement System**

#### New Measurement System
- **SP (Structural Preservation) metrics**: Adjacency, Order, Clustering preservation
- **SP-SSC independence validation**: Empirical demonstration of metric independence
- **Deterministic environment logging**: Automatic `env.txt` generation for all experiments

#### New Experiments (`experiments_sp/`)
- **I-2 series** (4 experiments): SP measurement system validation
  - sp00: Identity/isometry baseline
  - sp01: Full destruction benchmark
  - sp02: Topology rewire curves
  - sp03: Layout robustness
- **O-2 series** (4 experiments): Topological dominance
  - sp10: Metric invariance
  - sp11: Topology sensitivity
  - sp12: Topology vs metric comparison
  - sp13: Layout generalization
- **O-3 series** (3 experiments): SP-SSC independence
  - sp20: Coordinate noise
  - sp21: Semantic noise
  - sp22: Mixed noise grid
- **O-4 series** (2 experiments): Value-gated tradeoff
  - sp30: Lambda sweep (synthetic)
  - sp31: Lambda sweep (BERT)
- **Robustness series** (3 experiments):
  - sp40: Dimension/N robustness
  - sp41: Layout/topology robustness
  - sp42: k-NN parameter robustness

#### Demos
- **DR Evaluation Demo** (`demos/dr_evaluation/`):
  - Compare t-SNE, UMAP, PCA on MNIST
  - Visualize SP vs SSC tradeoffs
  - Publication-ready figures

#### Core Modules (`src/core_sp/`)
- `sp_metrics.py`: SP computation (adjacency/order/clustering)
- `ssc_wrapper.py`: SSC computation wrapper (v1 compatibility)
- `value_gate.py`: Value-gated coupling mechanism
- `generators.py`: Semantic embedding generation
- `topology_ops.py`: Topological disruption operations
- `metric_ops.py`: Metric transformation operations
- `deterministic.py`: Environment verification and manifest generation

#### Testing
- 101 new tests for SP measurement system
- Full cross-implementation validation
- Deterministic reproducibility verification

#### Workflows
- `run_experiments_sp.yml`: SP experiments execution (all/i2/o2/o3/o4/robust)
- `test_demo.yml`: DR evaluation demo testing

### Changed
- **Environment standardization**: Python 3.10.19, NumPy 1.24.3 across all experiments
- **Requirements**: Unified `requirements.txt` supporting both v1 (SSC) and v2 (SP) experiments
- Added `umap-learn` for dimensionality reduction demos

### Fixed
- Workflow Python version consistency (3.10 across all experiments)
- Environment documentation alignment

---

## [1.1.2] - 2025-11-16

### Fixed
- Environment documentation corrected to Python 3.10.19, NumPy 1.24.3
- Experiment data validated with corrected environment

---

## [1.1.1] - 2025-11-15

### Added
- Version-pinned dependencies in `requirements.txt`
- Deterministic execution environment specification

---

## [1.1.0] - 2025-11-14

### Added
- O-1 paper experiments (EXP-00 to EXP-13)
- BERT validation (SUP-14)
- Multilingual validation (SUP-15)
- Complete CI/CD pipeline

---

## [1.0.0] - 2025-11-06

### Added
- Initial release
- SSC measurement system (I-1)
- Core reproducibility framework

---

[2.0.0]: https://github.com/HIDEKI-SQ/intelligence-relativity/releases/tag/v2.0.0
[1.1.2]: https://github.com/HIDEKI-SQ/intelligence-relativity/releases/tag/v1.1.2
[1.1.1]: https://github.com/HIDEKI-SQ/intelligence-relativity/releases/tag/v1.1.1
[1.1.0]: https://github.com/HIDEKI-SQ/intelligence-relativity/releases/tag/v1.1.0
[1.0.0]: https://github.com/HIDEKI-SQ/intelligence-relativity/releases/tag/v1.0.0
