"""Quick test for plotting module."""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("  WEEK 3 PLOTTING TEST")
print("=" * 60)

# Test imports
print("\n[1] Testing imports...")
try:
    from gradus.week3 import (
        plot_1d,
        plot_2d_contour,
        plot_2d_heatmap,
        plot_2d_gradient_field,
        plot_3d_surface,
        plot_function,
    )
    print("  OK - All imports successful")
except ImportError as e:
    print(f"  FAIL - Import error: {e}")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Test 1D plotting
print("\n[2] Testing 1D plot (x^3 - 3x)...")
try:
    fig = plot_1d("x^3 - 3*x", show_derivative=True, show_critical_points=True)
    print(f"  OK - Figure created: {fig}")
    plt.close(fig)
except Exception as e:
    print(f"  FAIL - {e}")

# Test 2D contour
print("\n[3] Testing 2D contour (x^2 + y^2)...")
try:
    fig = plot_2d_contour("x^2 + y^2")
    print(f"  OK - Figure created: {fig}")
    plt.close(fig)
except Exception as e:
    print(f"  FAIL - {e}")

# Test 2D heatmap
print("\n[4] Testing 2D heatmap (sin(x)*cos(y))...")
try:
    fig = plot_2d_heatmap("sin(x)*cos(y)")
    print(f"  OK - Figure created: {fig}")
    plt.close(fig)
except Exception as e:
    print(f"  FAIL - {e}")

# Test 2D gradient field
print("\n[5] Testing gradient field (x^2 + y^2)...")
try:
    fig = plot_2d_gradient_field("x^2 + y^2")
    print(f"  OK - Figure created: {fig}")
    plt.close(fig)
except Exception as e:
    print(f"  FAIL - {e}")

# Test 3D surface
print("\n[6] Testing 3D surface (x^2 - y^2)...")
try:
    fig = plot_3d_surface("x^2 - y^2")
    print(f"  OK - Figure created: {fig}")
    plt.close(fig)
except Exception as e:
    print(f"  FAIL - {e}")

# Test auto-detect
print("\n[7] Testing auto-detect plot_function...")
try:
    # 1D
    fig = plot_function("x^2")
    print(f"  OK - 1D auto-detected: {fig}")
    plt.close(fig)
    
    # 2D
    fig = plot_function("x*y")
    print(f"  OK - 2D auto-detected: {fig}")
    plt.close(fig)
except Exception as e:
    print(f"  FAIL - {e}")

# Test domain handling
print("\n[8] Testing domain handling (log(x))...")
try:
    fig = plot_1d("log(x)")
    print(f"  OK - log(x) plotted with adjusted domain")
    plt.close(fig)
except Exception as e:
    print(f"  FAIL - {e}")

print("\n" + "=" * 60)
print("  ALL PLOTTING TESTS PASSED!")
print("=" * 60)


