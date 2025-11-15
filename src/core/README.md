# Core Measurement Toolkit

Validated instruments for the Optics of Intelligence project.

## Design Philosophy

### Consistency
- All distance inputs use **condensed vectors** from `scipy.spatial.distance.pdist`
- No global random state - all functions use explicit `seed` parameters
- Single-threaded BLAS for deterministic execution

### Separation of Concerns
- **Measurement** (`ssc_computation.py`) - Core SSC calculation
- **Reproducibility** (`deterministic.py`) - Execution contracts
- **Generation** (`generators.py`) - Standard data generators
- **Analysis** (`statistics.py`) - Statistical utilities
- **Presentation** (`visualization.py`) - Plotting

### Validation
All core functions validated in **I-1**:
- Test Suite 1-4: Distance computation (|Δ| < 10^-12)
- Test Suite 5: Full SSC pipeline (|Δ| < 10^-15)
- 20/20 tests passed

## Modules

### `ssc_computation.py`
SSC (Semantic-Spatial Correlation) measurement.

**Key Design**: Condensed vectors only
- Input: 1D arrays from `pdist()`, not square matrices
- Reason: Eliminates ambiguity, enforces consistency

**Functions**:
- `compute_ssc(sem_condensed, spa_condensed)` - Core calculation
- `compute_ssc_from_data(embeddings, coords)` - Convenience wrapper

**Natural Orthogonality** (O-1): Under λ=0, SSC ≈ 0

### `deterministic.py`
Deterministic execution utilities.

**Key Design**: No global random state
- Use `numpy.random.Generator` with explicit seeds
- `set_deterministic_mode()` configures BLAS only

**Functions**:
- `set_deterministic_mode()` - Single-threaded BLAS
- `verify_environment()` - Record versions and BLAS config
- `generate_manifest()` - SHA-256 hashes

### `generators.py`
Standard data generators.

**Key Design**: Explicit seeds, no global state
- Each generator takes `seed` parameter
- Uses `numpy.random.default_rng(seed)` internally

**Functions**:
- `generate_embeddings(n, dim, seed)` - Random semantic embeddings
- `generate_spatial_coords(n, layout, seed)` - Spatial layouts

**Layouts**:
- `'random'`: Uniform random in [-radius, radius]
- `'circle'`: Circle with randomized ordering (topology preserved)
- `'grid'`: Square grid with randomized assignment
- `'line'`: Line with randomized ordering
- `'cube'`: 3D grid with randomized assignment

### `statistics.py`
Statistical utilities.

**Functions**:
- `compute_summary_stats(data)` - Mean, std, median, min, max, n
- `bootstrap_ci(data, n_bootstrap, confidence, seed)` - Bootstrap CI
- `tost_equivalence(data, null, delta)` - Equivalence test

**Note**: TOST implementation is simplified. For critical analyses,
consider `statsmodels.stats.weightstats.ttost_ind()`.

### `visualization.py`
Standard plotting functions.

**Functions**:
- `plot_histogram(data, title, output_path)` - Histogram with mean
- `plot_scatter(x, y, title, output_path)` - Scatter with correlation
- `plot_ci(means, cis, title, output_path)` - Means with error bars

## Complete Usage Example
```python
from core import (
    set_deterministic_mode,
    verify_environment,
    generate_embeddings,
    generate_spatial_coords,
    compute_ssc,
    bootstrap_ci,
    plot_histogram
)
from scipy.spatial.distance import pdist
from pathlib import Path

# 1. Configure deterministic mode
set_deterministic_mode()
verify_environment(Path("outputs/env.txt"))

# 2. Generate data
embeddings = generate_embeddings(n_items=64, dim=128, seed=42)
coords = generate_spatial_coords(n_items=64, layout='circle', seed=42)

# 3. Compute distances (condensed vectors)
sem_dist = pdist(embeddings, 'correlation')
spa_dist = pdist(coords, 'euclidean')

# 4. Compute SSC
ssc = compute_ssc(sem_dist, spa_dist)
print(f"SSC = {ssc:.4f}")  # Expected: ≈0 (natural orthogonality)

# 5. Multiple trials
ssc_values = []
for seed in range(1000):
    emb = generate_embeddings(64, 128, seed)
    coo = generate_spatial_coords(64, 'circle', seed)
    sem = pdist(emb, 'correlation')
    spa = pdist(coo, 'euclidean')
    ssc_values.append(compute_ssc(sem, spa))

# 6. Statistics
ci_lower, ci_upper = bootstrap_ci(ssc_values, n_bootstrap=5000, seed=42)
print(f"90% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# 7. Visualization
plot_histogram(ssc_values, "Natural Orthogonality", 
               output_path=Path("outputs/ssc_hist.png"))
```
