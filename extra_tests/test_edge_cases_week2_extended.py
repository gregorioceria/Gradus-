"""Extended edge case tests for Week 2."""
from gradus import parse_expression
from gradus.week2 import (
    gradient, hessian, gradient_at_point, hessian_at_point,
    classify_critical_point, get_critical_point_info
)

print("=" * 70)
print("  EXTENDED EDGE CASES - WEEK 2")
print("=" * 70)

# =============================================================================
# TEST 1: Composite functions with domain issues
# =============================================================================
print("\n[1] COMPOSITE FUNCTIONS WITH DOMAIN ISSUES")
print("-" * 60)

# log(sin(x*y)) - sin can be <= 0
print("\nf(x,y) = log(sin(x*y))")
print("  Problem: sin(x*y) must be > 0")
test_cases = [
    {'x': 0.5, 'y': 1},      # sin(0.5) > 0, OK
    {'x': 3.14159, 'y': 1},  # sin(pi) ~ 0, log(0)
    {'x': 2, 'y': 2},        # sin(4) < 0, log(negative)
]
for point in test_cases:
    try:
        grad = gradient_at_point("log(sin(x*y))", point)
        print(f"  ({point['x']}, {point['y']}): grad = [{grad[0]:.4f}, {grad[1]:.4f}]")
    except Exception as e:
        print(f"  ({point['x']}, {point['y']}): ERROR - {type(e).__name__}")

# =============================================================================
# TEST 2: Square root with variables
# =============================================================================
print("\n[2] SQUARE ROOT - sqrt(x - y)")
print("-" * 60)

print("\nf(x,y) = sqrt(x - y)")
print("  Domain: x >= y")
test_cases = [
    {'x': 2, 'y': 1},      # x > y, OK
    {'x': 1, 'y': 1},      # x = y, sqrt(0) OK but gradient may have issues
    {'x': 0, 'y': 1},      # x < y, sqrt(negative)
]
for point in test_cases:
    try:
        grad = gradient_at_point("sqrt(x - y)", point)
        print(f"  ({point['x']}, {point['y']}): grad = [{grad[0]:.6f}, {grad[1]:.6f}]")
    except Exception as e:
        print(f"  ({point['x']}, {point['y']}): ERROR - {type(e).__name__}")

# =============================================================================
# TEST 3: Tangent function
# =============================================================================
print("\n[3] TANGENT - tan(x + y)")
print("-" * 60)

print("\nf(x,y) = tan(x + y)")
print("  Problem: x + y = pi/2 + n*pi -> undefined")
import math
test_cases = [
    {'x': 0, 'y': 0},                    # tan(0) = 0, OK
    {'x': 0.5, 'y': 0.5},                # tan(1) OK
    {'x': math.pi/4, 'y': math.pi/4},    # x + y = pi/2, UNDEFINED
    {'x': math.pi/2, 'y': 0},            # tan(pi/2), UNDEFINED
]
for point in test_cases:
    try:
        grad = gradient_at_point("tan(x + y)", point)
        print(f"  ({point['x']:.4f}, {point['y']:.4f}): grad = [{grad[0]:.4f}, {grad[1]:.4f}]")
    except Exception as e:
        print(f"  ({point['x']:.4f}, {point['y']:.4f}): ERROR - {type(e).__name__}")

# =============================================================================
# TEST 4: Fractional powers (cube root)
# =============================================================================
print("\n[4] FRACTIONAL POWERS - (x-y)^(1/3)")
print("-" * 60)

print("\nf(x,y) = (x - y)^(1/3)")
print("  Note: Cube root of negative is real, but SymPy returns complex")
test_cases = [
    {'x': 8, 'y': 0},    # (8)^(1/3) = 2
    {'x': 0, 'y': 0},    # (0)^(1/3) = 0, gradient undefined?
    {'x': 0, 'y': 8},    # (-8)^(1/3) = -2 (real) but SymPy gives complex
]
for point in test_cases:
    try:
        grad = gradient_at_point("(x - y)^(1/3)", point)
        print(f"  ({point['x']}, {point['y']}): grad = [{grad[0]:.6f}, {grad[1]:.6f}]")
    except Exception as e:
        print(f"  ({point['x']}, {point['y']}): ERROR - {type(e).__name__}")

# =============================================================================
# TEST 5: Absolute value
# =============================================================================
print("\n[5] ABSOLUTE VALUE - |x - y|")
print("-" * 60)

print("\nf(x,y) = |x - y|")
print("  Problem: Derivative undefined when x = y")
test_cases = [
    {'x': 2, 'y': 1},    # |1| = 1, derivative = sign(1) = 1
    {'x': 1, 'y': 2},    # |-1| = 1, derivative = sign(-1) = -1
    {'x': 1, 'y': 1},    # |0| = 0, derivative undefined (sign(0) = 0)
]
for point in test_cases:
    try:
        grad = gradient_at_point("Abs(x - y)", point)
        print(f"  ({point['x']}, {point['y']}): grad = [{grad[0]:.4f}, {grad[1]:.4f}]")
    except Exception as e:
        print(f"  ({point['x']}, {point['y']}): ERROR - {type(e).__name__}")

# =============================================================================
# TEST 6: Missing variables in point
# =============================================================================
print("\n[6] MISSING VARIABLES IN POINT")
print("-" * 60)

print("\nf(x,y,z) = x^2 + y^2 + z^2")
print("  What happens if we don't provide all variables?")
test_cases = [
    {'x': 1, 'y': 1, 'z': 1},   # All variables provided
    {'x': 1, 'y': 1},           # z missing
    {'x': 1},                    # y and z missing
]
for point in test_cases:
    try:
        grad = gradient_at_point("x^2 + y^2 + z^2", point)
        print(f"  {point}: grad = {grad}")
    except Exception as e:
        print(f"  {point}: ERROR - {type(e).__name__}: {e}")

# =============================================================================
# TEST 7: Numerical overflow
# =============================================================================
print("\n[7] NUMERICAL OVERFLOW - exp(x*y)")
print("-" * 60)

print("\nf(x,y) = exp(x*y)")
print("  Problem: Large x*y causes overflow")
test_cases = [
    {'x': 1, 'y': 1},      # exp(1) ~ 2.7
    {'x': 10, 'y': 10},    # exp(100) ~ 2.7e43
    {'x': 100, 'y': 100},  # exp(10000) -> overflow!
]
for point in test_cases:
    try:
        grad = gradient_at_point("exp(x*y)", point)
        print(f"  ({point['x']}, {point['y']}): grad = [{grad[0]:.4e}, {grad[1]:.4e}]")
    except Exception as e:
        print(f"  ({point['x']}, {point['y']}): ERROR - {type(e).__name__}")

# =============================================================================
# TEST 8: Singular Hessian (det = 0 but not all eigenvalues = 0)
# =============================================================================
print("\n[8] SINGULAR HESSIAN - det(H) = 0")
print("-" * 60)

print("\nf(x,y) = x^2 + xy")
H, vars_list = hessian("x^2 + x*y")
print(f"  Hessian: {H}")
H_at_0, _ = hessian_at_point("x^2 + x*y", {'x': 0, 'y': 0})
print(f"  H at (0,0): {H_at_0}")
classification, eigenvalues = classify_critical_point(H_at_0)
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Classification: {classification.upper()}")
det = H_at_0[0][0] * H_at_0[1][1] - H_at_0[0][1] * H_at_0[1][0]
print(f"  Determinant: {det}")

print("\nf(x,y) = (x + y)^2")
H, vars_list = hessian("(x + y)^2")
print(f"  Hessian: {H}")
H_at_0, _ = hessian_at_point("(x + y)^2", {'x': 0, 'y': 0})
print(f"  H at (0,0): {H_at_0}")
classification, eigenvalues = classify_critical_point(H_at_0)
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Classification: {classification.upper()}")

# =============================================================================
# TEST 9: Very small numbers (near zero)
# =============================================================================
print("\n[9] VERY SMALL NUMBERS - Near-zero precision")
print("-" * 60)

print("\nf(x,y) = 1/(x*y)")
test_cases = [
    {'x': 1, 'y': 1},
    {'x': 0.001, 'y': 0.001},
    {'x': 1e-10, 'y': 1e-10},
]
for point in test_cases:
    try:
        grad = gradient_at_point("1/(x*y)", point)
        print(f"  ({point['x']}, {point['y']}): grad = [{grad[0]:.4e}, {grad[1]:.4e}]")
    except Exception as e:
        print(f"  ({point['x']}, {point['y']}): ERROR - {type(e).__name__}")

# =============================================================================
# TEST 10: Complex expressions
# =============================================================================
print("\n[10] COMPLEX EXPRESSIONS")
print("-" * 60)

print("\nf(x,y) = sin(x)*cos(y)*exp(-x^2 - y^2)")
try:
    grad, _ = gradient("sin(x)*cos(y)*exp(-x^2 - y^2)")
    print(f"  Gradient (symbolic):")
    for i, g in enumerate(grad):
        print(f"    df/d{['x','y'][i]} = {g}")
    
    grad_val = gradient_at_point("sin(x)*cos(y)*exp(-x^2 - y^2)", {'x': 0, 'y': 0})
    print(f"  Gradient at (0,0): {grad_val}")
except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  SUMMARY OF POTENTIAL ISSUES")
print("=" * 70)
print("""
  1. log(sin(x*y)) - Composite domain restrictions
  2. sqrt(x-y)     - Square root of negative
  3. tan(x+y)      - Vertical asymptotes
  4. (x-y)^(1/3)   - SymPy returns complex for negative base
  5. |x-y|         - Non-differentiable at x=y
  6. Missing vars  - Silent failure or error?
  7. exp(x*y)      - Numerical overflow
  8. Singular H    - Degenerate critical points
  9. 1/(x*y)       - Numerical instability near zero
  10. Complex expr - Symbolic complexity
""")
print("=" * 70)


