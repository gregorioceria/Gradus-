"""
Test suite for Week 2 - Multivariable Derivatives.

Run with: python gradus/week2/tests/test_week2.py
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from gradus.week2 import (
    partial_derivative,
    all_partial_derivatives,
    gradient,
    hessian,
    gradient_at_point,
    hessian_at_point,
    mixed_partial,
)
from gradus.week2.multi_var import TooManyVariablesError, NoVariablesError
from sympy import Symbol, Matrix


def test_case(description: str, test_func):
    """Run a test case and print result."""
    try:
        result = test_func()
        if result:
            print(f"  [OK] {description}")
            return True
        else:
            print(f"  [FAIL] {description} - Test returned False")
            return False
    except Exception as e:
        print(f"  [FAIL] {description} - {type(e).__name__}: {e}")
        return False


def run_partial_derivative_tests():
    """Test partial derivative computation."""
    print("\n" + "=" * 60)
    print("  PARTIAL DERIVATIVE TESTS")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    print("\n[1] Basic Partial Derivatives")
    print("-" * 40)
    
    def test_pd_x():
        result = partial_derivative("x^2 + y^2", "x")
        return str(result) == "2*x"
    
    def test_pd_y():
        result = partial_derivative("x^2 + y^2", "y")
        return str(result) == "2*y"
    
    def test_pd_xy():
        result = partial_derivative("x*y", "x")
        return str(result) == "y"
    
    def test_pd_second():
        result = partial_derivative("x^3 + y^3", "x", order=2)
        return str(result) == "6*x"
    
    total += 4
    if test_case("d/dx(x^2 + y^2) = 2x", test_pd_x): passed += 1
    if test_case("d/dy(x^2 + y^2) = 2y", test_pd_y): passed += 1
    if test_case("d/dx(x*y) = y", test_pd_xy): passed += 1
    if test_case("d2/dx2(x^3 + y^3) = 6x", test_pd_second): passed += 1
    
    print("\n[2] All Partial Derivatives")
    print("-" * 40)
    
    def test_all_pd():
        result = all_partial_derivatives("x^2 + y^2 + xy")
        x, y = Symbol('x'), Symbol('y')
        return (str(result[x]) == "2*x + y" and 
                str(result[y]) == "x + 2*y")
    
    total += 1
    if test_case("all_partial_derivatives(x^2 + y^2 + xy)", test_all_pd): passed += 1
    
    print("\n[3] Mixed Partial Derivatives")
    print("-" * 40)
    
    def test_mixed():
        result = mixed_partial("x^2 * y^3", ['x', 'y'])
        return str(result) == "6*x*y**2"
    
    total += 1
    if test_case("d2/dydx(x^2 * y^3) = 6xy^2", test_mixed): passed += 1
    
    return passed, total


def run_gradient_tests():
    """Test gradient computation."""
    print("\n" + "=" * 60)
    print("  GRADIENT TESTS")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    print("\n[1] Gradient Vector")
    print("-" * 40)
    
    def test_grad_simple():
        grad, vars_list = gradient("x^2 + y^2")
        return (str(grad[0]) == "2*x" and 
                str(grad[1]) == "2*y")
    
    def test_grad_3var():
        grad, vars_list = gradient("x^2 + y^2 + z^2")
        return len(grad) == 3
    
    total += 2
    if test_case("grad(x^2 + y^2) = [2x, 2y]", test_grad_simple): passed += 1
    if test_case("grad has 3 components for 3 vars", test_grad_3var): passed += 1
    
    print("\n[2] Gradient at Point")
    print("-" * 40)
    
    def test_grad_at_point():
        result = gradient_at_point("x^2 + y^2", {'x': 1, 'y': 2})
        return abs(result[0] - 2.0) < 0.001 and abs(result[1] - 4.0) < 0.001
    
    total += 1
    if test_case("grad(x^2+y^2) at (1,2) = [2, 4]", test_grad_at_point): passed += 1
    
    return passed, total


def run_hessian_tests():
    """Test Hessian computation."""
    print("\n" + "=" * 60)
    print("  HESSIAN TESTS")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    print("\n[1] Hessian Matrix")
    print("-" * 40)
    
    def test_hessian_simple():
        H, vars_list = hessian("x^2 + y^2")
        # Should be [[2, 0], [0, 2]]
        return (H[0,0] == 2 and H[0,1] == 0 and 
                H[1,0] == 0 and H[1,1] == 2)
    
    def test_hessian_mixed():
        H, vars_list = hessian("x^2 + y^2 + xy")
        # Should be [[2, 1], [1, 2]]
        return (H[0,0] == 2 and H[0,1] == 1 and 
                H[1,0] == 1 and H[1,1] == 2)
    
    def test_hessian_symmetric():
        H, _ = hessian("x^3*y^2 + sin(x*y)")
        # Hessian should be symmetric
        return H[0,1] == H[1,0]
    
    total += 3
    if test_case("H(x^2 + y^2) = [[2,0],[0,2]]", test_hessian_simple): passed += 1
    if test_case("H(x^2 + y^2 + xy) = [[2,1],[1,2]]", test_hessian_mixed): passed += 1
    if test_case("Hessian is symmetric", test_hessian_symmetric): passed += 1
    
    print("\n[2] Hessian at Point")
    print("-" * 40)
    
    def test_hessian_at_point():
        H, _ = hessian_at_point("x^2 + y^2 + xy", {'x': 1, 'y': 1})
        return (abs(H[0][0] - 2.0) < 0.001 and 
                abs(H[0][1] - 1.0) < 0.001)
    
    total += 1
    if test_case("H at (1,1) correct", test_hessian_at_point): passed += 1
    
    print("\n[3] 3x3 Hessian")
    print("-" * 40)
    
    def test_hessian_3x3():
        H, vars_list = hessian("x^2 + y^2 + z^2")
        return H.shape == (3, 3)
    
    total += 1
    if test_case("3-var Hessian is 3x3", test_hessian_3x3): passed += 1
    
    return passed, total


def run_error_tests():
    """Test error handling."""
    print("\n" + "=" * 60)
    print("  ERROR HANDLING TESTS")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    print("\n[1] Too Many Variables")
    print("-" * 40)
    
    def test_too_many_vars():
        try:
            gradient("a + b + c + d + f + g + h")  # 7 variables (avoid 'e' - Euler)
            return False
        except TooManyVariablesError:
            return True
    
    total += 1
    if test_case("7 variables raises TooManyVariablesError", test_too_many_vars): passed += 1
    
    print("\n[2] No Variables")
    print("-" * 40)
    
    def test_no_vars():
        try:
            gradient("5")  # No variables
            return False
        except NoVariablesError:
            return True
    
    total += 1
    if test_case("Constant raises NoVariablesError", test_no_vars): passed += 1
    
    return passed, total


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  GRADUS WEEK 2 - TEST SUITE")
    print("=" * 60)
    
    pd_passed, pd_total = run_partial_derivative_tests()
    grad_passed, grad_total = run_gradient_tests()
    hess_passed, hess_total = run_hessian_tests()
    err_passed, err_total = run_error_tests()
    
    total_passed = pd_passed + grad_passed + hess_passed + err_passed
    total_tests = pd_total + grad_total + hess_total + err_total
    
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    print(f"\n  Partial Deriv Tests: {pd_passed}/{pd_total}")
    print(f"  Gradient Tests:      {grad_passed}/{grad_total}")
    print(f"  Hessian Tests:       {hess_passed}/{hess_total}")
    print(f"  Error Tests:         {err_passed}/{err_total}")
    print(f"  {'-' * 30}")
    print(f"  TOTAL:               {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n  ALL TESTS PASSED!")
    else:
        print(f"\n  {total_tests - total_passed} test(s) failed.")
    
    print("=" * 60 + "\n")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

