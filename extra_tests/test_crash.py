"""
Test crash cases for plotting.
"""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gradus.week3 import plot_1d, plot_2d_heatmap, plot_3d_surface, plot_function

def test_case(name, func, *args, **kwargs):
    """Test a single case and report result."""
    print(f"\n[TEST] {name}")
    print(f"       Args: {args}, Kwargs: {kwargs}")
    try:
        fig = func(*args, **kwargs)
        plt.close(fig)
        print("       PASS - No crash")
        return True
    except Exception as e:
        print(f"       CRASH: {type(e).__name__}: {e}")
        return False

print("=" * 60)
print("  CRASH TESTING")
print("=" * 60)

crashes = []

# =============================================================================
# 1D POTENTIAL CRASHES
# =============================================================================
print("\n--- 1D Tests ---")

# Test 1: All NaN function
if not test_case("sqrt(-x) - All NaN in default range", plot_1d, "sqrt(-x)"):
    crashes.append("sqrt(-x)")

# Test 2: Extreme growth
if not test_case("exp(x^2) - Extreme growth", plot_1d, "exp(x^2)"):
    crashes.append("exp(x^2)")

# Test 3: Division everywhere undefined
if not test_case("1/sin(x) - Multiple singularities", plot_1d, "1/sin(x)"):
    crashes.append("1/sin(x)")

# Test 4: Complex result
if not test_case("x^(1/3) with negative x", plot_1d, "x^(1/3)", x_range=(-5, 5)):
    crashes.append("x^(1/3)")

# Test 5: Very nested function
if not test_case("log(log(log(x)))", plot_1d, "log(log(log(x)))"):
    crashes.append("log(log(log(x)))")

# Test 6: Factorial-like growth
if not test_case("x^x", plot_1d, "x^x", x_range=(0.1, 5)):
    crashes.append("x^x")

# Test 7: asin outside domain
if not test_case("asin(x) full range", plot_1d, "asin(x)", x_range=(-5, 5)):
    crashes.append("asin(x)")

# =============================================================================
# 2D POTENTIAL CRASHES
# =============================================================================
print("\n--- 2D Tests ---")

# Test 8: All complex results
if not test_case("sqrt(-(x^2+y^2))", plot_2d_heatmap, "sqrt(-(x^2+y^2))"):
    crashes.append("sqrt(-(x^2+y^2))")

# Test 9: Division by complex expression
if not test_case("1/(x-y)", plot_2d_heatmap, "1/(x-y)"):
    crashes.append("1/(x-y)")

# Test 10: Extreme 2D growth
if not test_case("exp(x*y)", plot_2d_heatmap, "exp(x*y)"):
    crashes.append("exp(x*y)")

# Test 11: log of potentially negative
if not test_case("log(x-y)", plot_2d_heatmap, "log(x-y)"):
    crashes.append("log(x-y)")

# =============================================================================
# 3D POTENTIAL CRASHES
# =============================================================================
print("\n--- 3D Tests ---")

# Test 12: 3D with singularity at origin
if not test_case("1/sqrt(x^2+y^2)", plot_3d_surface, "1/sqrt(x^2+y^2)"):
    crashes.append("1/sqrt(x^2+y^2)")

# Test 13: 3D all NaN
if not test_case("sqrt(-x-y)", plot_3d_surface, "sqrt(-x-y)"):
    crashes.append("sqrt(-x-y)")

# Test 14: Very complex 3D
if not test_case("tan(x)*tan(y)", plot_3d_surface, "tan(x)*tan(y)"):
    crashes.append("tan(x)*tan(y)")

# =============================================================================
# AUTO-DETECT EDGE CASES
# =============================================================================
print("\n--- Auto-detect Tests ---")

# Test 15: Constant function
if not test_case("5 (constant)", plot_function, "5"):
    crashes.append("5")

# Test 16: Empty/weird input
try:
    if not test_case("x + ", plot_function, "x + "):
        crashes.append("x + ")
except:
    print("       CRASH on parsing (expected)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

if crashes:
    print(f"\nCRASHED ({len(crashes)}):")
    for c in crashes:
        print(f"  - {c}")
else:
    print("\nNo crashes! All tests passed.")


