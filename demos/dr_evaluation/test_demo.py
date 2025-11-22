"""Quick test for DR evaluation demo (no actual execution)."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from demos.dr_evaluation import load_data
        print("  ✅ load_data")
    except ImportError as e:
        print(f"  ❌ load_data: {e}")
        return False
    
    try:
        from demos.dr_evaluation import apply_dr
        print("  ✅ apply_dr")
    except ImportError as e:
        print(f"  ❌ apply_dr: {e}")
        return False
    
    try:
        from demos.dr_evaluation import compute_metrics
        print("  ✅ compute_metrics")
    except ImportError as e:
        print(f"  ❌ compute_metrics: {e}")
        return False
    
    try:
        from demos.dr_evaluation import visualize_results
        print("  ✅ visualize_results")
    except ImportError as e:
        print(f"  ❌ visualize_results: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True

def test_core_sp():
    """Test that core_sp modules are available."""
    print("\nTesting core_sp availability...")
    
    try:
        from src.core_sp import compute_sp_total, compute_ssc
        print("  ✅ core_sp modules available")
        return True
    except ImportError as e:
        print(f"  ❌ core_sp: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("DR Evaluation Demo - Quick Test")
    print("="*60)
    print()
    
    success = True
    success = test_imports() and success
    success = test_core_sp() and success
    
    print()
    print("="*60)
    if success:
        print("✅ All tests passed!")
        print("\nReady to run: python demos/dr_evaluation/run_dr_demo.py")
    else:
        print("❌ Some tests failed")
        print("\nPlease check dependencies and imports")
    print("="*60)
