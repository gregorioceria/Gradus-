"""
Extreme crash test cases.
"""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import signal

from gradus.week3 import plot_1d, plot_2d_heatmap, plot_3d_surface, plot_function

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout!")

def test_case(name, func, *args, timeout_sec=30, **kwargs):
    """Test with timeout."""
    print(f"\n[TEST] {name}")
    try:
        # Windows doesn't support signal.alarm, so we just run
        fig = func(*args, **kwargs)
        plt.close(fig)
        print("       PASS")
        return "pass"
    except MemoryError as e:
        print(f"       MEMORY ERROR: {e}")
        return "memory"
    except RecursionError as e:
        print(f"       RECURSION ERROR: {e}")
        return "recursion"
    except Exception as e:
        print(f"       ERROR ({type(e).__name__}): {str(e)[:100]}")
        return "error"

print("=" * 60)
print("  EXTREME CRASH TESTING")
print("=" * 60)

results = {"pass": 0, "memory": 0, "recursion": 0, "error": 0}

# =============================================================================
# MEMORY STRESS
# =============================================================================
print("\n--- Memory Stress Tests ---")

# Very high resolution
r = test_case("1D with 10000 points", plot_1d, "sin(x)", num_points=10000)
results[r] += 1

r = test_case("2D with 500x500 points", plot_2d_heatmap, "x^2+y^2", num_points=500)
results[r] += 1

r = test_case("3D with 300x300 points", plot_3d_surface, "x^2+y^2", num_points=300)
results[r] += 1

# =============================================================================
# COMPLEX EXPRESSIONS
# =============================================================================
print("\n--- Complex Expression Tests ---")

# Deeply nested
r = test_case("sin(sin(sin(sin(sin(x)))))", plot_1d, "sin(sin(sin(sin(sin(x)))))")
results[r] += 1

# Long polynomial
poly = "+".join([f"{i}*x^{i}" for i in range(1, 20)])
r = test_case(f"Long polynomial ({len(poly)} chars)", plot_1d, poly)
results[r] += 1

# Product of many terms
prod = "*".join(["(x+1)" for _ in range(10)])
r = test_case("Product of 10 terms", plot_1d, prod)
results[r] += 1

# =============================================================================
# NUMERICAL EDGE CASES
# =============================================================================
print("\n--- Numerical Edge Cases ---")

# Very small range
r = test_case("Very small range [0.0001, 0.0002]", plot_1d, "x^2", x_range=(0.0001, 0.0002))
results[r] += 1

# Very large range
r = test_case("Very large range [-1000000, 1000000]", plot_1d, "x", x_range=(-1000000, 1000000))
results[r] += 1

# Negative range for sqrt
r = test_case("sqrt with all-negative range", plot_1d, "sqrt(x)", x_range=(-10, -1))
results[r] += 1

# =============================================================================
# SPECIAL CHARACTERS
# =============================================================================
print("\n--- Special Character Tests ---")

try:
    r = test_case("Unicode variable", plot_1d, "alpha + 1")
    results[r] += 1
except:
    print("       ERROR on unicode")
    results["error"] += 1

# =============================================================================
# PATHOLOGICAL 2D/3D
# =============================================================================
print("\n--- Pathological 2D/3D ---")

# All undefined
r = test_case("sqrt(-(x^2+y^2)) - all complex", plot_3d_surface, "sqrt(-(x^2+y^2))")
results[r] += 1

# Infinite at many points
r = test_case("1/(sin(x)*sin(y)) - many singularities", plot_3d_surface, "1/(sin(x)*sin(y))", x_range=(-6, 6), y_range=(-6, 6))
results[r] += 1

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"\nPassed:     {results['pass']}")
print(f"Errors:     {results['error']}")
print(f"Memory:     {results['memory']}")
print(f"Recursion:  {results['recursion']}")


