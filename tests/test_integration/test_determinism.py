"""Determinism integration tests for v2.0.0.

Validates the deterministic reproducibility standard:
    - std = 0.00 across multiple runs
    - Bit-for-bit reproducibility
    - Cross-platform consistency (where applicable)
    - Seed isolation

This test suite parallels v1.1.2's 16/16 test standard.
"""

import pytest
import numpy as np
import json
from pathlib import Path
import shutil

from src.core_sp.sp_metrics import compute_sp_total
from src.core_sp.topology_ops import edge_rewire, permute_coords, random_relayout
from src.core_sp.metric_ops import rotate_2d, scale_2d, shear_2d, add_coord_noise
from src.core_sp.generators import generate_semantic_embeddings, add_semantic_noise
from src.core_sp.ssc_wrapper import compute_ssc
from src.core_sp.value_gate import apply_value_gate


@pytest.fixture
def sample_coords():
    """Standard grid coordinates."""
    xs = np.linspace(-1, 1, 8)
    ys = np.linspace(-1, 1, 8)
    xv, yv = np.meshgrid(xs, ys)
    return np.stack([xv.ravel(), yv.ravel()], axis=1)


class TestCoreSPDeterminism:
    """Determinism tests for core_sp module."""
    
    def test_sp_metrics_std_zero(self, sample_coords):
        """Test: SP computation has std=0.00 across runs."""
        results = []
        for _ in range(10):
            sp = compute_sp_total(sample_coords, sample_coords, layout_type="grid")
            results.append(sp)
        
        std = np.std(results)
        assert std == 0.0, f"SP std should be 0.00, got {std}"
    
    def test_topology_ops_std_zero(self, sample_coords):
        """Test: Topology operations std=0.00 with same seed."""
        results = []
        for _ in range(10):
            rng = np.random.default_rng(42)
            coords = permute_coords(sample_coords, rng=rng, p=0.5)
            sp = compute_sp_total(sample_coords, coords, layout_type="grid")
            results.append(sp)
        
        std = np.std(results)
        assert std == 0.0, f"Topology ops std should be 0.00, got {std}"
    
    def test_metric_ops_std_zero(self, sample_coords):
        """Test: Metric operations std=0.00."""
        results = []
        for _ in range(10):
            coords = shear_2d(sample_coords, k=0.5)
            sp = compute_sp_total(sample_coords, coords, layout_type="grid")
            results.append(sp)
        
        std = np.std(results)
        assert std == 0.0, f"Metric ops std should be 0.00, got {std}"
    
    def test_generators_std_zero(self):
        """Test: Generators std=0.00 with same seed."""
        results = []
        for _ in range(10):
            rng = np.random.default_rng(42)
            emb = generate_semantic_embeddings(64, 128, rng)
            results.append(emb[0, 0])  # Check first element
        
        std = np.std(results)
        assert std == 0.0, f"Generators std should be 0.00, got {std}"
    
    def test_ssc_std_zero(self):
        """Test: SSC computation std=0.00."""
        rng = np.random.default_rng(42)
        embeddings = generate_semantic_embeddings(64, 128, rng)
        coords = rng.uniform(-1, 1, (64, 2))
        
        results = []
        for _ in range(10):
            ssc = compute_ssc(embeddings, coords)
            results.append(ssc)
        
        std = np.std(results)
        assert std == 0.0, f"SSC std should be 0.00, got {std}"
    
    def test_value_gate_std_zero(self):
        """Test: Value gate std=0.00 with same seed."""
        rng = np.random.default_rng(42)
        embeddings = generate_semantic_embeddings(20, 100, rng)
        base_coords = rng.uniform(-1, 1, (20, 2))
        
        results = []
        for _ in range(10):
            coords = apply_value_gate(base_coords, embeddings, lam=0.5, seed=42)
            results.append(coords[0, 0])  # Check first element
        
        std = np.std(results)
        assert std == 0.0, f"Value gate std should be 0.00, got {std}"


class TestSeedIsolation:
    """Test that different seeds produce different results."""
    
    def test_generators_seed_isolation(self):
        """Test: Different seeds → different embeddings."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        
        emb1 = generate_semantic_embeddings(64, 128, rng1)
        emb2 = generate_semantic_embeddings(64, 128, rng2)
        
        assert not np.allclose(emb1, emb2), "Different seeds must produce different results"
    
    def test_topology_ops_seed_isolation(self, sample_coords):
        """Test: Different seeds → different permutations."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        
        coords1 = permute_coords(sample_coords, rng=rng1, p=0.5)
        coords2 = permute_coords(sample_coords, rng=rng2, p=0.5)
        
        assert not np.allclose(coords1, coords2), "Different seeds must produce different results"
    
    def test_value_gate_seed_isolation(self):
        """Test: Different seeds → different arrangements."""
        rng = np.random.default_rng(42)
        embeddings = generate_semantic_embeddings(20, 100, rng)
        base_coords = rng.uniform(-1, 1, (20, 2))
        
        coords1 = apply_value_gate(base_coords, embeddings, lam=0.5, seed=42)
        coords2 = apply_value_gate(base_coords, embeddings, lam=0.5, seed=99)
        
        assert not np.allclose(coords1, coords2), "Different seeds must produce different results"


class TestBitForBitReproducibility:
    """Test bit-for-bit reproducibility (hash equality)."""
    
    def test_sp_computation_hash_equality(self, sample_coords):
        """Test: Identical hash across runs."""
        import hashlib
        
        hashes = []
        for _ in range(5):
            sp = compute_sp_total(sample_coords, sample_coords, layout_type="grid")
            hash_val = hashlib.sha256(np.array([sp]).tobytes()).hexdigest()
            hashes.append(hash_val)
        
        assert len(set(hashes)) == 1, "Hashes should be identical (bit-for-bit)"
    
    def test_embeddings_hash_equality(self):
        """Test: Identical embeddings hash."""
        import hashlib
        
        hashes = []
        for _ in range(5):
            rng = np.random.default_rng(42)
            emb = generate_semantic_embeddings(64, 128, rng)
            hash_val = hashlib.sha256(emb.tobytes()).hexdigest()
            hashes.append(hash_val)
        
        assert len(set(hashes)) == 1, "Hashes should be identical (bit-for-bit)"


class TestExperimentDeterminism:
    """Integration tests for full experiments."""
    
    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Temporary output directory."""
        output_dir = tmp_path / "test_outputs_sp"
        output_dir.mkdir(parents=True, exist_ok=True)
        yield output_dir
        if output_dir.exists():
            shutil.rmtree(output_dir)
    
    def test_sp00_determinism(self, temp_output_dir):
        """Test: SP-00 produces identical results."""
        from src.experiments_sp.i2_sp_instrument.sp00_identity_isometry import run_sp00_identity_isometry
        
        out_dir1 = temp_output_dir / "run1"
        out_dir2 = temp_output_dir / "run2"
        
        run_sp00_identity_isometry(n_trials=3, seed=42, out_dir=out_dir1)
        run_sp00_identity_isometry(n_trials=3, seed=42, out_dir=out_dir2)
        
        with open(out_dir1 / "sp00_identity_isometry_raw.json") as f:
            data1 = json.load(f)
        with open(out_dir2 / "sp00_identity_isometry_raw.json") as f:
            data2 = json.load(f)
        
        sp_vals1 = [r["sp"] for r in data1["records"]]
        sp_vals2 = [r["sp"] for r in data2["records"]]
        
        assert np.allclose(sp_vals1, sp_vals2, atol=0, rtol=0), \
            "SP-00 results must be bit-for-bit identical"
    
    def test_sp30_determinism(self, temp_output_dir):
        """Test: SP-30 produces identical results."""
        from src.experiments_sp.o4_value_gate_tradeoff_sp.sp30_lambda_sweep_synth import run_sp30_lambda_sweep_synth
        
        out_dir1 = temp_output_dir / "run1"
        out_dir2 = temp_output_dir / "run2"
        
        run_sp30_lambda_sweep_synth(n_trials=3, seed=600, out_dir=out_dir1)
        run_sp30_lambda_sweep_synth(n_trials=3, seed=600, out_dir=out_dir2)
        
        with open(out_dir1 / "sp30_lambda_sweep_synth_raw.json") as f:
            data1 = json.load(f)
        with open(out_dir2 / "sp30_lambda_sweep_synth_raw.json") as f:
            data2 = json.load(f)
        
        ssc1 = [r["ssc"] for r in data1["records"]]
        ssc2 = [r["ssc"] for r in data2["records"]]
        
        assert np.allclose(ssc1, ssc2, atol=0, rtol=0), \
            "SP-30 SSC results must be bit-for-bit identical"


class TestDeterminismSummary:
    """Summary test: 16/16 style validation."""
    
    def test_all_core_operations_deterministic(self, sample_coords):
        """Test: All core operations pass determinism check (16/16 style)."""
        rng = np.random.default_rng(42)
        embeddings = generate_semantic_embeddings(64, 128, rng)
        coords_rand = rng.uniform(-1, 1, (64, 2))
        
        operations = {
            "sp_identity": lambda: compute_sp_total(sample_coords, sample_coords, "grid"),
            "sp_rotation": lambda: compute_sp_total(sample_coords, rotate_2d(sample_coords, np.pi/4), "grid"),
            "sp_shear": lambda: compute_sp_total(sample_coords, shear_2d(sample_coords, 0.5), "grid"),
            "permute": lambda: permute_coords(sample_coords, np.random.default_rng(42), 0.5)[0, 0],
            "generate": lambda: generate_semantic_embeddings(10, 20, np.random.default_rng(42))[0, 0],
            "ssc": lambda: compute_ssc(embeddings, coords_rand),
            "value_gate": lambda: apply_value_gate(coords_rand[:20], embeddings[:20], 0.5, seed=42)[0, 0],
        }
        
        passed = 0
        failed = []
        
        for name, op in operations.items():
            results = [op() for _ in range(5)]
            std = np.std(results)
            if std == 0.0:
                passed += 1
            else:
                failed.append((name, std))
        
        total = len(operations)
        print(f"\n✅ Determinism: {passed}/{total} tests passed (std=0.00)")
        
        if failed:
            print("❌ Failed tests:")
            for name, std in failed:
                print(f"   {name}: std={std}")
        
        assert passed == total, f"Expected {total}/{total} deterministic, got {passed}/{total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
