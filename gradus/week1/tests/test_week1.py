"""
Test suite for Week 1 - Parsing and Single Variable Derivatives.

Run with: python -m pytest gradus/week1/tests/test_week1.py -v
Or directly: python gradus/week1/tests/test_week1.py
"""

import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from gradus.week1 import (
    parse_expression,
    validate_expression,
    first_derivative,
    second_derivative,
    third_derivative,
    all_derivatives,
    derivative_at_point,
    ParsingError,
    EmptyExpressionError,
    UnbalancedParenthesesError,
    EmptyFunctionArgumentError,
    UnsupportedFunctionError,
    MalformedOperatorError,
    DivisionByZeroError,
)


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


def run_parsing_tests():
    """Test parsing edge cases."""
    print("\n" + "=" * 60)
    print("  PARSING TESTS")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # Empty expressions
    print("\n[1] Empty Expression Tests")
    print("-" * 40)
    
    def test_empty():
        try:
            parse_expression("")
            return False
        except EmptyExpressionError:
            return True
    
    total += 1
    if test_case("Empty string raises error", test_empty):
        passed += 1
    
    # Parentheses
    print("\n[2] Parentheses Tests")
    print("-" * 40)
    
    def test_unbalanced():
        try:
            parse_expression("(x + 1")
            return False
        except UnbalancedParenthesesError:
            return True
    
    def test_balanced():
        return parse_expression("((x + 1) * (y + 2))") is not None
    
    total += 2
    if test_case("Unbalanced parentheses detected", test_unbalanced):
        passed += 1
    if test_case("Balanced parentheses OK", test_balanced):
        passed += 1
    
    # Empty function arguments
    print("\n[3] Function Argument Tests")
    print("-" * 40)
    
    def test_empty_sin():
        try:
            parse_expression("sin()")
            return False
        except EmptyFunctionArgumentError:
            return True
    
    def test_valid_sin():
        return parse_expression("sin(x)") is not None
    
    total += 2
    if test_case("sin() - empty argument error", test_empty_sin):
        passed += 1
    if test_case("sin(x) - valid", test_valid_sin):
        passed += 1
    
    # Unsupported functions
    print("\n[4] Function Support Tests")
    print("-" * 40)
    
    def test_unsupported():
        try:
            parse_expression("myfunc(x)")
            return False
        except UnsupportedFunctionError:
            return True
    
    def test_all_supported():
        funcs = ["sin(x)", "cos(x)", "tan(x)", "exp(x)", "log(x)", "sqrt(x)"]
        for f in funcs:
            if parse_expression(f) is None:
                return False
        return True
    
    total += 2
    if test_case("Unsupported function detected", test_unsupported):
        passed += 1
    if test_case("All standard functions work", test_all_supported):
        passed += 1
    
    # Malformed operators
    print("\n[5] Operator Tests")
    print("-" * 40)
    
    def test_triple_plus():
        try:
            parse_expression("x +++ y")
            return False
        except MalformedOperatorError:
            return True
    
    def test_valid_power():
        return parse_expression("x ** 2") is not None
    
    total += 2
    if test_case("Triple plus error", test_triple_plus):
        passed += 1
    if test_case("Power operator OK", test_valid_power):
        passed += 1
    
    # Division by zero
    print("\n[6] Division by Zero Tests")
    print("-" * 40)
    
    def test_div_zero():
        try:
            parse_expression("x / 0")
            return False
        except DivisionByZeroError:
            return True
    
    def test_div_var():
        return parse_expression("x / y") is not None
    
    total += 2
    if test_case("Division by zero detected", test_div_zero):
        passed += 1
    if test_case("Division by variable OK", test_div_var):
        passed += 1
    
    # Power notation
    print("\n[7] Notation Tests")
    print("-" * 40)
    
    def test_caret():
        return str(parse_expression("x^2")) == "x**2"
    
    def test_ln():
        return "log" in str(parse_expression("ln(x)"))
    
    total += 2
    if test_case("Caret converts to power", test_caret):
        passed += 1
    if test_case("ln converts to log", test_ln):
        passed += 1
    
    return passed, total


def run_derivative_tests():
    """Test derivative computations."""
    print("\n" + "=" * 60)
    print("  DERIVATIVE TESTS")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # Basic polynomials
    print("\n[1] Polynomial Derivatives")
    print("-" * 40)
    
    def test_x2():
        return str(first_derivative("x^2")) == "2*x"
    
    def test_x3():
        return str(first_derivative("x^3")) == "3*x**2"
    
    def test_x3_second():
        return str(second_derivative("x^3")) == "6*x"
    
    def test_x3_third():
        return str(third_derivative("x^3")) == "6"
    
    total += 4
    if test_case("d/dx(x^2) = 2x", test_x2):
        passed += 1
    if test_case("d/dx(x^3) = 3x^2", test_x3):
        passed += 1
    if test_case("d2/dx2(x^3) = 6x", test_x3_second):
        passed += 1
    if test_case("d3/dx3(x^3) = 6", test_x3_third):
        passed += 1
    
    # Trigonometric
    print("\n[2] Trigonometric Derivatives")
    print("-" * 40)
    
    def test_sin():
        return str(first_derivative("sin(x)")) == "cos(x)"
    
    def test_cos():
        return str(first_derivative("cos(x)")) == "-sin(x)"
    
    total += 2
    if test_case("d/dx(sin(x)) = cos(x)", test_sin):
        passed += 1
    if test_case("d/dx(cos(x)) = -sin(x)", test_cos):
        passed += 1
    
    # Exponential
    print("\n[3] Exponential Derivatives")
    print("-" * 40)
    
    def test_exp():
        return str(first_derivative("exp(x)")) == "exp(x)"
    
    def test_log():
        return str(first_derivative("log(x)")) == "1/x"
    
    total += 2
    if test_case("d/dx(exp(x)) = exp(x)", test_exp):
        passed += 1
    if test_case("d/dx(log(x)) = 1/x", test_log):
        passed += 1
    
    # All derivatives function
    print("\n[4] All Derivatives Function")
    print("-" * 40)
    
    def test_all_derivs():
        result = all_derivatives("x^4")
        return (
            str(result['f']) == "x**4" and
            str(result['f_prime']) == "4*x**3" and
            str(result['f_double_prime']) == "12*x**2" and
            str(result['f_triple_prime']) == "24*x"
        )
    
    total += 1
    if test_case("all_derivatives(x^4) correct", test_all_derivs):
        passed += 1
    
    # Derivative at point
    print("\n[5] Numerical Evaluation")
    print("-" * 40)
    
    def test_at_point():
        result = derivative_at_point("x^2", point=3, order=1)
        return abs(result - 6.0) < 0.001
    
    total += 1
    if test_case("f'(3) where f=x^2 equals 6", test_at_point):
        passed += 1
    
    return passed, total


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  GRADUS WEEK 1 - TEST SUITE")
    print("=" * 60)
    
    parsing_passed, parsing_total = run_parsing_tests()
    deriv_passed, deriv_total = run_derivative_tests()
    
    total_passed = parsing_passed + deriv_passed
    total_tests = parsing_total + deriv_total
    
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    print(f"\n  Parsing Tests:    {parsing_passed}/{parsing_total}")
    print(f"  Derivative Tests: {deriv_passed}/{deriv_total}")
    print(f"  {'-' * 30}")
    print(f"  TOTAL:            {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n  ALL TESTS PASSED!")
    else:
        print(f"\n  {total_tests - total_passed} test(s) failed.")
    
    print("=" * 60 + "\n")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

