"""
Domain checking module for mathematical expressions.

This module provides utilities to check if evaluation points
are within the valid domain of functions.
"""

import math
from typing import Optional, Tuple, List
from sympy import (
    Symbol, log, tan, asin, acos, Abs, sign,
    oo, zoo, nan, pi, Float, Rational, Pow
)
from sympy.functions.elementary.miscellaneous import sqrt as sympy_sqrt


# =============================================================================
# CONFIGURATION
# =============================================================================

# Maximum exponent allowed (soft and hard limits)
MAX_EXPONENT_SOFT = 100  # Warning
MAX_EXPONENT_HARD = 1000  # Error

# Maximum coefficient
MAX_COEFFICIENT = 10**15

# Computation timeout in seconds
COMPUTATION_TIMEOUT = 15


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class DomainError(ValueError):
    """Raised when a value is outside the domain of a function."""
    pass


class LargeNumberWarning(Warning):
    """Warning for potentially large computations."""
    pass


class LargeNumberError(ValueError):
    """Error for excessively large numbers."""
    pass


# =============================================================================
# DOMAIN CHECKING FUNCTIONS
# =============================================================================

def check_log_domain(x: float, func_name: str = "log") -> Tuple[bool, Optional[str]]:
    """
    Check if x is in the domain of logarithm (x > 0).
    
    Returns:
        (is_valid, error_message)
    """
    if x <= 0:
        return (False, f"{func_name}(x) is undefined for x <= 0. Got x = {x}")
    return (True, None)


def check_sqrt_domain(x: float) -> Tuple[bool, Optional[str]]:
    """
    Check if x is in the domain of square root (x >= 0 for real numbers).
    
    Returns:
        (is_valid, error_message)
    """
    if x < 0:
        return (False, f"sqrt(x) is undefined for x < 0 in real numbers. Got x = {x}")
    return (True, None)


def check_division_domain(x: float) -> Tuple[bool, Optional[str]]:
    """
    Check if x is valid for division (x != 0).
    
    Returns:
        (is_valid, error_message)
    """
    if x == 0:
        return (False, "Division by zero: x = 0")
    return (True, None)


def check_tan_domain(x: float) -> Tuple[bool, Optional[str]]:
    """
    Check if x is in the domain of tangent (x != pi/2 + k*pi).
    
    Returns:
        (is_valid, error_message)
    """
    pi_val = math.pi
    
    # Check distance from nearest pi/2 + k*pi
    # tan is undefined at pi/2 + k*pi for integer k
    # That's equivalent to x = pi/2, 3pi/2, 5pi/2, ... and -pi/2, -3pi/2, ...
    
    # Normalize: how far is x from the nearest odd multiple of pi/2?
    # x mod pi tells us position within a period
    remainder = x % pi_val
    
    # Distance from pi/2 (the singularity within [0, pi])
    dist_to_singularity = abs(remainder - pi_val/2)
    
    # Use a tolerance for floating point comparison
    tolerance = 1e-6
    
    if dist_to_singularity < tolerance:
        return (False, f"tan(x) is undefined at x = pi/2 + k*pi. Got x = {x}")
    return (True, None)


def check_asin_domain(x: float) -> Tuple[bool, Optional[str]]:
    """
    Check if x is in the domain of arcsine (-1 <= x <= 1).
    
    Returns:
        (is_valid, error_message)
    """
    if x < -1 or x > 1:
        return (False, f"asin(x) requires -1 <= x <= 1. Got x = {x}")
    return (True, None)


def check_acos_domain(x: float) -> Tuple[bool, Optional[str]]:
    """
    Check if x is in the domain of arccosine (-1 <= x <= 1).
    
    Returns:
        (is_valid, error_message)
    """
    if x < -1 or x > 1:
        return (False, f"acos(x) requires -1 <= x <= 1. Got x = {x}")
    return (True, None)


def check_abs_derivative_domain(x: float) -> Tuple[bool, Optional[str]]:
    """
    Check domain for derivative of absolute value (undefined at x = 0).
    
    Returns:
        (is_valid, warning_message)
    """
    if x == 0:
        return (False, "Warning: derivative of |x| is undefined at x = 0")
    return (True, None)


# =============================================================================
# EXPRESSION DOMAIN ANALYSIS
# =============================================================================

def get_domain_restrictions(expr) -> List[str]:
    """
    Analyze an expression and return a list of domain restrictions.
    
    Args:
        expr: SymPy expression
        
    Returns:
        List of domain restriction descriptions
    """
    restrictions = []
    
    # Check for log
    if expr.has(log):
        restrictions.append("log(x): requires x > 0")
    
    # Check for roots (sqrt, 4th root, 6th root, etc.)
    root_restrictions_added = set()
    for root_order, base, requires_non_neg in _find_root_expressions(expr):
        if requires_non_neg and root_order not in root_restrictions_added:
            if root_order == 2:
                restrictions.append("sqrt(x): requires x >= 0")
            else:
                restrictions.append(f"{root_order}th root: requires x >= 0")
            root_restrictions_added.add(root_order)
    
    # Check for tan
    if expr.has(tan):
        restrictions.append("tan(x): undefined at x = pi/2 + k*pi")
    
    # Check for asin/acos
    if expr.has(asin):
        restrictions.append("asin(x): requires -1 <= x <= 1")
    if expr.has(acos):
        restrictions.append("acos(x): requires -1 <= x <= 1")
    
    # Check for abs (derivative issue)
    if expr.has(Abs):
        restrictions.append("|x|: derivative undefined at x = 0")
    
    # Check for division (1/x terms)
    for subexpr in expr.atoms(Pow):
        if subexpr.exp.is_negative:
            restrictions.append("1/x^n: undefined at x = 0")
            break
    
    return restrictions


def _find_functions(expr, func_type):
    """Find all instances of a function type in an expression."""
    from sympy import preorder_traversal
    results = []
    for subexpr in preorder_traversal(expr):
        if hasattr(subexpr, 'func') and subexpr.func == func_type:
            results.append(subexpr)
    return results


def _find_root_expressions(expr):
    """
    Find all root expressions, including:
    - sqrt(x) and x^(1/2)
    - x^(1/n) for any n (nth root)
    
    Returns list of (root_order, base, requires_non_negative)
    - root_order: the denominator (2 for sqrt, 3 for cube root, etc.)
    - base: the expression under the root
    - requires_non_negative: True if even root (needs x >= 0)
    """
    from sympy import preorder_traversal, sqrt, Rational, Integer
    results = []
    
    for subexpr in preorder_traversal(expr):
        # Check for explicit sqrt function
        if hasattr(subexpr, 'func') and subexpr.func == sqrt:
            results.append((2, subexpr.args[0], True))  # sqrt requires x >= 0
        # Check for x^(p/q) which is a root
        elif hasattr(subexpr, 'is_Pow') and subexpr.is_Pow:
            base, exp = subexpr.as_base_exp()
            
            # Check if exponent is a fraction with numerator 1 (simple root)
            # or any fraction (fractional power)
            if hasattr(exp, 'is_Rational') and exp.is_Rational and exp.q != 1:
                # exp.q is the denominator (root order)
                # exp.p is the numerator
                root_order = exp.q
                # Even roots require non-negative base
                requires_non_neg = (root_order % 2 == 0)
                results.append((root_order, base, requires_non_neg))
            elif hasattr(exp, 'is_number') and exp.is_number:
                try:
                    exp_float = float(exp)
                    # Check if it's a fractional exponent like 0.5, 0.25, 0.333...
                    if 0 < exp_float < 1:
                        # Approximate root order
                        root_order = int(round(1 / exp_float))
                        requires_non_neg = (root_order % 2 == 0)
                        results.append((root_order, base, requires_non_neg))
                except:
                    pass
    
    return results


def check_root_domain(x: float, root_order: int) -> Tuple[bool, Optional[str]]:
    """
    Check if x is in the domain of nth root.
    
    Even roots (2, 4, 6, ...): require x >= 0
    Odd roots (3, 5, 7, ...): defined for all real x
    
    Returns:
        (is_valid, error_message)
    """
    if root_order % 2 == 0 and x < 0:
        if root_order == 2:
            return (False, f"sqrt(x) is undefined for x < 0 in real numbers. Got x = {x}")
        else:
            return (False, f"{root_order}th root is undefined for x < 0 in real numbers. Got x = {x}")
    return (True, None)


def check_evaluation_point(expr, var, point: float) -> Tuple[bool, Optional[str]]:
    """
    Check if a point is valid for evaluating an expression.
    
    Args:
        expr: SymPy expression
        var: Variable symbol
        point: Value to check
        
    Returns:
        (is_valid, error_message or None)
    """
    from sympy import sqrt
    
    # Substitute and check for issues
    try:
        # Check log domain
        for subexpr in _find_functions(expr, log):
            arg = subexpr.args[0]
            arg_val = float(arg.subs(var, point))
            valid, msg = check_log_domain(arg_val)
            if not valid:
                return (False, msg)
        
        # Check root domains (sqrt, cube root, 4th root, etc.)
        for root_order, base, requires_non_neg in _find_root_expressions(expr):
            if requires_non_neg:  # Only check even roots
                base_val = float(base.subs(var, point))
                valid, msg = check_root_domain(base_val, root_order)
                if not valid:
                    return (False, msg)
        
        # Check tan domain
        for subexpr in _find_functions(expr, tan):
            arg = subexpr.args[0]
            arg_val = float(arg.subs(var, point))
            valid, msg = check_tan_domain(arg_val)
            if not valid:
                return (False, msg)
        
        # Check asin/acos domain
        for subexpr in _find_functions(expr, asin):
            arg = subexpr.args[0]
            arg_val = float(arg.subs(var, point))
            valid, msg = check_asin_domain(arg_val)
            if not valid:
                return (False, msg)
        
        for subexpr in _find_functions(expr, acos):
            arg = subexpr.args[0]
            arg_val = float(arg.subs(var, point))
            valid, msg = check_acos_domain(arg_val)
            if not valid:
                return (False, msg)
        
        # Try actual evaluation to catch division by zero etc.
        result = expr.subs(var, point)
        if result.has(oo, -oo, zoo, nan):
            return (False, f"Expression is undefined at x = {point}")
        
        # Try converting to float
        float(result)
        
        return (True, None)
        
    except (ZeroDivisionError, ValueError) as e:
        return (False, f"Cannot evaluate at x = {point}: {e}")
    except Exception as e:
        return (False, f"Evaluation error at x = {point}: {e}")


# =============================================================================
# LARGE NUMBER CHECKS
# =============================================================================

def check_exponent_size(expr) -> Tuple[bool, Optional[str], bool]:
    """
    Check if expression has excessively large exponents.
    
    Returns:
        (is_ok, message, is_warning_only)
        - is_ok: False if hard limit exceeded
        - message: Warning or error message
        - is_warning_only: True if just a warning, False if error
    """
    from sympy import Pow, Integer, Float
    
    for subexpr in expr.atoms(Pow):
        exp = subexpr.exp
        
        # Try to get numeric value of exponent
        try:
            if exp.is_number:
                exp_val = abs(float(exp))
                
                if exp_val > MAX_EXPONENT_HARD:
                    return (
                        False, 
                        f"Exponent too large: {exp_val} (max: {MAX_EXPONENT_HARD})",
                        False
                    )
                elif exp_val > MAX_EXPONENT_SOFT:
                    return (
                        True,
                        f"Warning: large exponent ({exp_val}) may cause slow computation",
                        True
                    )
        except (TypeError, ValueError):
            pass  # Non-numeric exponent, skip
    
    return (True, None, True)


def check_coefficient_size(expr) -> Tuple[bool, Optional[str]]:
    """
    Check if expression has excessively large coefficients.
    
    Returns:
        (is_ok, message)
    """
    from sympy import Integer, Float, Rational
    
    for num in expr.atoms(Integer, Float, Rational):
        try:
            val = abs(float(num))
            if val > MAX_COEFFICIENT:
                return (False, f"Coefficient too large: {val} (max: {MAX_COEFFICIENT})")
        except (TypeError, ValueError, OverflowError):
            return (False, f"Number too large to process: {num}")
    
    return (True, None)


def validate_expression_size(expr) -> Tuple[bool, List[str]]:
    """
    Validate that an expression doesn't have excessively large numbers.
    
    Returns:
        (is_valid, list_of_warnings_or_errors)
    """
    messages = []
    is_valid = True
    
    # Check exponents
    ok, msg, is_warning = check_exponent_size(expr)
    if msg:
        messages.append(msg)
        if not ok:
            is_valid = False
    
    # Check coefficients
    ok, msg = check_coefficient_size(expr)
    if msg:
        messages.append(msg)
        if not ok:
            is_valid = False
    
    return (is_valid, messages)

