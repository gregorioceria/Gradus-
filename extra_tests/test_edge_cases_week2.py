"""Test edge cases for Week 2."""
from gradus import parse_expression
from gradus.week2 import (
    gradient, hessian, gradient_at_point, hessian_at_point,
    classify_critical_point, get_critical_point_info
)
from sympy import Matrix, N


# Using classify_critical_point from gradus.week2


print("=" * 65)
print("  EDGE CASES WEEK 2")
print("=" * 65)

# =============================================================================
# TEST 1: Hessian with eigenvalue = 0 (degenerate critical points)
# =============================================================================
print("\n[1] HESSIAN WITH EIGENVALUE = 0 (Degenerate Cases)")
print("-" * 55)

# Case 1a: f(x,y) = x^3 - Hessian is [[6x, 0], [0, 0]]
# At x=0: H = [[0, 0], [0, 0]] -> all eigenvalues = 0
print("\nf(x,y) = x^3")
H, vars_list = hessian("x^3 + 0*y")  # Add 0*y to make it 2-var
print(f"  Hessian (symbolic): {H}")

# Evaluate at origin
H_at_0, _ = hessian_at_point("x^3 + 0*y", {'x': 0, 'y': 0})
print(f"  Hessian at (0,0): {H_at_0}")

classification, eigenvalues = classify_critical_point(H_at_0)
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Classification: {classification.upper()}")
print(f"  -> Cannot determine if min/max/saddle using second derivative test!")

# Case 1b: f(x,y) = x^2*y^2 - saddle point at origin but degenerate
print("\nf(x,y) = x^2 * y^2")
H, vars_list = hessian("x^2 * y^2")
print(f"  Hessian (symbolic): {H}")

H_at_0, _ = hessian_at_point("x^2 * y^2", {'x': 0, 'y': 0})
print(f"  Hessian at (0,0): {H_at_0}")

classification, eigenvalues = classify_critical_point(H_at_0)
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Classification: {classification.upper()}")

# Case 1c: "Monkey saddle" f(x,y) = x^3 - 3xy^2
print("\nf(x,y) = x^3 - 3xy^2 (Monkey Saddle)")
H, vars_list = hessian("x^3 - 3*x*y^2")
print(f"  Hessian (symbolic): {H}")

H_at_0, _ = hessian_at_point("x^3 - 3*x*y^2", {'x': 0, 'y': 0})
print(f"  Hessian at (0,0): {H_at_0}")

classification, eigenvalues = classify_critical_point(H_at_0)
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Classification: {classification.upper()}")

# =============================================================================
# TEST 1d: Regular cases (for comparison)
# =============================================================================
print("\n--- Regular cases (non-degenerate) ---")

print("\nf(x,y) = x^2 + y^2 (LOCAL MINIMUM at origin)")
H_at_0, _ = hessian_at_point("x^2 + y^2", {'x': 0, 'y': 0})
classification, eigenvalues = classify_critical_point(H_at_0)
print(f"  Hessian: {H_at_0}")
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Classification: {classification.upper()}")

print("\nf(x,y) = -x^2 - y^2 (LOCAL MAXIMUM at origin)")
H_at_0, _ = hessian_at_point("-x^2 - y^2", {'x': 0, 'y': 0})
classification, eigenvalues = classify_critical_point(H_at_0)
print(f"  Hessian: {H_at_0}")
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Classification: {classification.upper()}")

print("\nf(x,y) = x^2 - y^2 (SADDLE POINT at origin)")
H_at_0, _ = hessian_at_point("x^2 - y^2", {'x': 0, 'y': 0})
classification, eigenvalues = classify_critical_point(H_at_0)
print(f"  Hessian: {H_at_0}")
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Classification: {classification.upper()}")

# =============================================================================
# TEST 2: Domain issues - log(x-y) when x close to y
# =============================================================================
print("\n" + "=" * 65)
print("[2] DOMAIN ISSUES - log(x-y)")
print("-" * 55)

expr = parse_expression("log(x - y)")
print(f"\nf(x,y) = log(x - y)")
print(f"  Domain: x - y > 0, i.e., x > y")

# Test gradient
grad, vars_list = gradient(expr)
print(f"\n  Gradient: {grad}")

# Try evaluation at various points
test_points = [
    {'x': 2, 'y': 1},      # x > y, OK
    {'x': 1.001, 'y': 1},  # x slightly > y, should work
    {'x': 1, 'y': 1},      # x = y, UNDEFINED (log(0))
    {'x': 0.5, 'y': 1},    # x < y, UNDEFINED (log of negative)
]

print("\n  Evaluating gradient at points:")
for point in test_points:
    x_val, y_val = point['x'], point['y']
    diff = x_val - y_val
    try:
        grad_val = gradient_at_point(expr, point)
        print(f"    ({x_val}, {y_val}): x-y = {diff:>8.4f} -> grad = [{grad_val[0]:.4f}, {grad_val[1]:.4f}]")
    except Exception as e:
        error_type = "log(0)" if diff == 0 else "log(negative)" if diff < 0 else str(e)
        print(f"    ({x_val}, {y_val}): x-y = {diff:>8.4f} -> ERROR: {error_type}")

# =============================================================================
# TEST 3: sqrt(x^2 + y^2) - always defined except gradient at origin
# =============================================================================
print("\n" + "=" * 65)
print("[3] sqrt(x^2 + y^2) - Gradient undefined at origin")
print("-" * 55)

expr = parse_expression("sqrt(x^2 + y^2)")
print(f"\nf(x,y) = sqrt(x^2 + y^2)")

grad, vars_list = gradient(expr)
print(f"  Gradient: {grad}")

test_points = [
    {'x': 1, 'y': 0},
    {'x': 0, 'y': 1},
    {'x': 0.001, 'y': 0.001},
    {'x': 0, 'y': 0},  # Origin - gradient is 0/0
]

print("\n  Evaluating gradient at points:")
for point in test_points:
    x_val, y_val = point['x'], point['y']
    try:
        grad_val = gradient_at_point(expr, point)
        print(f"    ({x_val}, {y_val}): grad = [{grad_val[0]:.6f}, {grad_val[1]:.6f}]")
    except Exception as e:
        print(f"    ({x_val}, {y_val}): ERROR - {e}")

# =============================================================================
# TEST 4: 1/(x-y) - division by zero when x=y
# =============================================================================
print("\n" + "=" * 65)
print("[4] 1/(x-y) - Division by zero when x=y")
print("-" * 55)

expr = parse_expression("1/(x - y)")
print(f"\nf(x,y) = 1/(x - y)")

grad, vars_list = gradient(expr)
print(f"  Gradient: {grad}")

test_points = [
    {'x': 2, 'y': 1},
    {'x': 1.01, 'y': 1},
    {'x': 1, 'y': 1},  # x = y, division by zero
]

print("\n  Evaluating gradient at points:")
for point in test_points:
    x_val, y_val = point['x'], point['y']
    try:
        grad_val = gradient_at_point(expr, point)
        print(f"    ({x_val}, {y_val}): grad = [{grad_val[0]:.6f}, {grad_val[1]:.6f}]")
    except Exception as e:
        print(f"    ({x_val}, {y_val}): ERROR - Division by zero")

# =============================================================================
# TEST 5: Using get_critical_point_info API
# =============================================================================
print("\n" + "=" * 65)
print("[5] get_critical_point_info API")
print("-" * 55)

print("\nAnalyzing f(x,y) = x^2 + y^2 at origin:")
info = get_critical_point_info("x^2 + y^2", {'x': 0, 'y': 0})
print(f"  Is critical point: {info['is_critical_point']}")
print(f"  Gradient: {info['gradient']}")
print(f"  Hessian: {info['hessian']}")
print(f"  Eigenvalues: {info['eigenvalues']}")
print(f"  Classification: {info['classification'].upper()}")

print("\nAnalyzing f(x,y) = x^3 - 3xy^2 at origin (DEGENERATE):")
info = get_critical_point_info("x^3 - 3*x*y^2", {'x': 0, 'y': 0})
print(f"  Is critical point: {info['is_critical_point']}")
print(f"  Gradient: {info['gradient']}")
print(f"  Classification: {info['classification'].upper()}")
if info['classification'] == 'degenerate':
    print(f"  NOTE: Second derivative test is INCONCLUSIVE")

print("\nAnalyzing f(x,y) = x^2 - y^2 at (1, 1) - NOT a critical point:")
info = get_critical_point_info("x^2 - y^2", {'x': 1, 'y': 1})
print(f"  Is critical point: {info['is_critical_point']}")
print(f"  Gradient: {info['gradient']}")
if info.get('warning'):
    print(f"  WARNING: {info['warning']}")

print("\n" + "=" * 65)
print("  EDGE CASE TESTS COMPLETED")
print("=" * 65)
