"""
Multivariable derivative computation module.

This module provides functions to compute:
- Partial derivatives
- Gradient vectors
- Hessian matrices

Supports up to 6 variables.
"""

from sympy import (
    diff, Symbol, simplify, Matrix, symbols,
    Rational, Float, Integer, N
)
from typing import List, Dict, Tuple, Optional, Union
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week1.parsing import parse_expression, get_variables


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_VARIABLES = 6


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class TooManyVariablesError(ValueError):
    """Raised when expression has more than MAX_VARIABLES."""
    pass


class NoVariablesError(ValueError):
    """Raised when expression has no variables."""
    pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_sorted_variables(expr) -> List[Symbol]:
    """
    Get variables from expression sorted alphabetically.
    
    Args:
        expr: SymPy expression or string
        
    Returns:
        List of Symbol objects sorted by name
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    variables = get_variables(expr)
    return sorted(list(variables), key=lambda x: str(x))


def _validate_variable_count(expr) -> List[Symbol]:
    """
    Validate that expression has between 1 and MAX_VARIABLES variables.
    
    Returns:
        Sorted list of variables
        
    Raises:
        NoVariablesError: If no variables
        TooManyVariablesError: If too many variables
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    variables = _get_sorted_variables(expr)
    
    if len(variables) == 0:
        raise NoVariablesError(
            "Expression has no variables. Cannot compute partial derivatives."
        )
    
    if len(variables) > MAX_VARIABLES:
        raise TooManyVariablesError(
            f"Expression has {len(variables)} variables, maximum is {MAX_VARIABLES}. "
            f"Variables found: {', '.join(str(v) for v in variables)}"
        )
    
    return variables


# =============================================================================
# PARTIAL DERIVATIVES
# =============================================================================

def partial_derivative(expr, var, order: int = 1, simplify_result: bool = True):
    """
    Compute partial derivative of expression with respect to a variable.
    
    Args:
        expr: SymPy expression or string
        var: Variable to differentiate with respect to (Symbol or string)
        order: Order of derivative (default 1)
        simplify_result: Whether to simplify the result
        
    Returns:
        SymPy expression representing the partial derivative
        
    Examples:
        >>> partial_derivative("x^2 + y^2", "x")
        2*x
        >>> partial_derivative("x^2 * y^3", "y", order=2)
        6*x**2*y
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    if isinstance(var, str):
        var = Symbol(var)
    
    result = diff(expr, var, order)
    
    if simplify_result:
        result = simplify(result)
    
    return result


def all_partial_derivatives(expr, order: int = 1, simplify_result: bool = True) -> Dict[Symbol, any]:
    """
    Compute all first-order partial derivatives.
    
    Args:
        expr: SymPy expression or string
        order: Order of derivatives (default 1)
        simplify_result: Whether to simplify results
        
    Returns:
        Dictionary mapping each variable to its partial derivative
        
    Examples:
        >>> all_partial_derivatives("x^2 + y^2 + xy")
        {x: 2*x + y, y: 2*y + x}
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    variables = _validate_variable_count(expr)
    
    result = {}
    for var in variables:
        result[var] = partial_derivative(expr, var, order, simplify_result)
    
    return result


def mixed_partial(expr, vars_sequence: List, simplify_result: bool = True):
    """
    Compute mixed partial derivative.
    
    Args:
        expr: SymPy expression or string
        vars_sequence: List of variables to differentiate with respect to, in order
                      e.g., ['x', 'y'] computes ∂²f/∂y∂x
        simplify_result: Whether to simplify result
        
    Returns:
        SymPy expression
        
    Examples:
        >>> mixed_partial("x^2 * y^3", ['x', 'y'])  # ∂²f/∂y∂x
        6*x*y**2
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    result = expr
    for var in vars_sequence:
        if isinstance(var, str):
            var = Symbol(var)
        result = diff(result, var)
    
    if simplify_result:
        result = simplify(result)
    
    return result


# =============================================================================
# GRADIENT
# =============================================================================

def gradient(expr, simplify_result: bool = True) -> Tuple[List, List[Symbol]]:
    """
    Compute the gradient vector of a scalar function.
    
    The gradient is the vector of all first partial derivatives:
    ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    
    Args:
        expr: SymPy expression or string
        simplify_result: Whether to simplify results
        
    Returns:
        Tuple of (gradient_list, variables_list)
        - gradient_list: List of partial derivatives
        - variables_list: List of variables (in order)
        
    Examples:
        >>> grad, vars = gradient("x^2 + y^2 + xy")
        >>> grad
        [2*x + y, 2*y + x]
        >>> vars
        [x, y]
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    variables = _validate_variable_count(expr)
    
    grad = []
    for var in variables:
        pd = partial_derivative(expr, var, 1, simplify_result)
        grad.append(pd)
    
    return (grad, variables)


def _check_multivar_domain(expr, point: Dict[str, float]) -> tuple:
    """
    Check if a point is in the domain of a multivariable expression.
    
    Returns:
        (is_valid, error_message or None)
    """
    from sympy import log, sqrt, tan, oo, zoo, nan, I, Float
    import math
    
    # Build substitution dict
    subs_dict = {Symbol(k): v for k, v in point.items()}
    
    # Evaluate the expression
    try:
        result = expr.subs(subs_dict)
        result_n = N(result)
        
        # Check for infinity, undefined, or complex (symbolic)
        if result.has(oo, -oo, zoo, nan):
            return (False, "Expression is undefined at this point")
        
        # Check if result is complex
        if result_n.has(I) or (hasattr(result_n, 'is_real') and result_n.is_real == False):
            return (False, "Expression results in complex value (outside real domain)")
        
        # Try to convert to float - will fail for complex
        try:
            val = float(result_n)
            # Check for numerical infinity or NaN
            if math.isinf(val):
                return (False, "Expression results in infinity (overflow)")
            if math.isnan(val):
                return (False, "Expression results in NaN (undefined)")
        except (TypeError, ValueError):
            return (False, "Expression results in non-real value")
        except OverflowError:
            return (False, "Expression results in overflow")
        
        return (True, None)
        
    except RecursionError:
        return (False, "Expression causes recursion error (may involve Abs)")
    except Exception as e:
        return (False, f"Cannot evaluate: {e}")


def gradient_at_point(expr, point: Dict[str, float], simplify_result: bool = True) -> List[float]:
    """
    Evaluate the gradient at a specific point.
    
    Args:
        expr: SymPy expression or string
        point: Dictionary mapping variable names to values
               e.g., {'x': 1, 'y': 2}
        simplify_result: Whether to simplify before evaluation
        
    Returns:
        List of numerical values
        
    Raises:
        ValueError: If point is outside the domain
        
    Examples:
        >>> gradient_at_point("x^2 + y^2", {'x': 1, 'y': 2})
        [2.0, 4.0]
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    # Check domain first
    is_valid, error_msg = _check_multivar_domain(expr, point)
    if not is_valid:
        raise ValueError(error_msg)
    
    grad, variables = gradient(expr, simplify_result)
    
    # Build substitution dict with Symbol keys
    subs_dict = {}
    for var_name, value in point.items():
        subs_dict[Symbol(var_name)] = value
    
    import math
    
    result = []
    for i, component in enumerate(grad):
        try:
            val = component.subs(subs_dict)
            val_n = N(val)
        except RecursionError:
            var_name = str(variables[i])
            raise ValueError(f"Gradient component df/d{var_name} causes recursion error (may involve Abs)")
        
        # Check for undefined gradient component (symbolic)
        from sympy import oo, zoo, nan, I
        if val.has(oo, -oo, zoo, nan) or val_n.has(I):
            var_name = str(variables[i])
            raise ValueError(f"Gradient component df/d{var_name} is undefined at this point")
        
        try:
            float_val = float(val_n)
            # Check for numerical infinity or NaN
            if math.isinf(float_val):
                var_name = str(variables[i])
                raise ValueError(f"Gradient component df/d{var_name} is infinite at this point")
            if math.isnan(float_val):
                var_name = str(variables[i])
                raise ValueError(f"Gradient component df/d{var_name} is NaN at this point")
            # Check for extremely large values (near asymptote)
            # This catches cases like tan(pi/2) where float precision doesn't give exactly infinity
            # Using 1e100 as threshold - allows large but finite values like exp(100) ~ 2.7e43
            if abs(float_val) > 1e100:
                var_name = str(variables[i])
                raise ValueError(f"Gradient component df/d{var_name} is extremely large (near asymptote)")
            result.append(float_val)
        except (TypeError, ValueError) as e:
            if "Gradient component" in str(e):
                raise
            var_name = str(variables[i])
            raise ValueError(f"Gradient component df/d{var_name} is non-real at this point")
        except OverflowError:
            var_name = str(variables[i])
            raise ValueError(f"Gradient component df/d{var_name} causes overflow")
    
    return result


# =============================================================================
# HESSIAN MATRIX
# =============================================================================

def hessian(expr, simplify_result: bool = True) -> Tuple[Matrix, List[Symbol]]:
    """
    Compute the Hessian matrix of a scalar function.
    
    The Hessian is the matrix of all second partial derivatives:
    H[i,j] = ∂²f/∂xᵢ∂xⱼ
    
    Args:
        expr: SymPy expression or string
        simplify_result: Whether to simplify results
        
    Returns:
        Tuple of (hessian_matrix, variables_list)
        - hessian_matrix: SymPy Matrix object
        - variables_list: List of variables (defines row/column order)
        
    Examples:
        >>> H, vars = hessian("x^2 + y^2 + xy")
        >>> H
        Matrix([[2, 1], [1, 2]])
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    variables = _validate_variable_count(expr)
    n = len(variables)
    
    # Build the Hessian matrix
    H = []
    for i, var_i in enumerate(variables):
        row = []
        for j, var_j in enumerate(variables):
            # Compute ∂²f/∂xᵢ∂xⱼ
            second_deriv = diff(diff(expr, var_i), var_j)
            if simplify_result:
                second_deriv = simplify(second_deriv)
            row.append(second_deriv)
        H.append(row)
    
    return (Matrix(H), variables)


def hessian_at_point(expr, point: Dict[str, float], simplify_result: bool = True) -> Tuple[List[List[float]], List[Symbol]]:
    """
    Evaluate the Hessian matrix at a specific point.
    
    Args:
        expr: SymPy expression or string
        point: Dictionary mapping variable names to values
        simplify_result: Whether to simplify before evaluation
        
    Returns:
        Tuple of (numerical_matrix, variables_list)
        - numerical_matrix: 2D list of floats
        - variables_list: List of variables
        
    Raises:
        ValueError: If point is outside the domain
        
    Examples:
        >>> H, vars = hessian_at_point("x^2 + y^2 + xy", {'x': 1, 'y': 2})
        >>> H
        [[2.0, 1.0], [1.0, 2.0]]
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    # Check domain first
    is_valid, error_msg = _check_multivar_domain(expr, point)
    if not is_valid:
        raise ValueError(error_msg)
    
    H_symbolic, variables = hessian(expr, simplify_result)
    
    # Build substitution dict with Symbol keys
    subs_dict = {}
    for var_name, value in point.items():
        subs_dict[Symbol(var_name)] = value
    
    # Evaluate each element
    from sympy import oo, zoo, nan, I
    n = len(variables)
    H_numerical = []
    for i in range(n):
        row = []
        for j in range(n):
            val = H_symbolic[i, j].subs(subs_dict)
            val_n = N(val)
            
            # Check for undefined values
            if val.has(oo, -oo, zoo, nan) or val_n.has(I):
                raise ValueError(
                    f"Hessian element H[{i},{j}] (d2f/d{variables[i]}d{variables[j]}) "
                    f"is undefined at this point"
                )
            
            try:
                row.append(float(val_n))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Hessian element H[{i},{j}] is non-real at this point"
                )
        H_numerical.append(row)
    
    return (H_numerical, variables)


# =============================================================================
# CRITICAL POINT CLASSIFICATION
# =============================================================================

def classify_critical_point(H_numerical: List[List[float]], tolerance: float = 1e-10) -> Tuple[str, List[float]]:
    """
    Classify a critical point based on Hessian eigenvalues.
    
    Uses the second derivative test to determine the nature of a critical point.
    
    Args:
        H_numerical: Numerical Hessian matrix (2D list of floats)
        tolerance: Values smaller than this are considered zero
        
    Returns:
        Tuple of (classification, eigenvalues)
        - classification: One of:
            'local_minimum': All eigenvalues > 0 (positive definite)
            'local_maximum': All eigenvalues < 0 (negative definite)
            'saddle_point': Mixed signs (indefinite)
            'degenerate': At least one eigenvalue = 0 (inconclusive)
        - eigenvalues: List of eigenvalues as floats
        
    Examples:
        >>> H = [[2.0, 0.0], [0.0, 2.0]]  # Hessian of x^2 + y^2 at origin
        >>> classify_critical_point(H)
        ('local_minimum', [2.0, 2.0])
        
        >>> H = [[2.0, 0.0], [0.0, -2.0]]  # Hessian of x^2 - y^2 at origin
        >>> classify_critical_point(H)
        ('saddle_point', [2.0, -2.0])
        
        >>> H = [[0.0, 0.0], [0.0, 0.0]]  # Degenerate case
        >>> classify_critical_point(H)
        ('degenerate', [0.0, 0.0])
    """
    H_matrix = Matrix(H_numerical)
    eigenvalues_dict = H_matrix.eigenvals()
    
    # Convert eigenvalues to floats
    eigenvalues = []
    for ev, multiplicity in eigenvalues_dict.items():
        ev_float = float(N(ev))
        # Add eigenvalue according to its multiplicity
        for _ in range(multiplicity):
            eigenvalues.append(ev_float)
    
    # Sort eigenvalues for consistent output
    eigenvalues.sort(reverse=True)
    
    # Check for zeros (degenerate case)
    has_zero = any(abs(ev) < tolerance for ev in eigenvalues)
    
    if has_zero:
        return ('degenerate', eigenvalues)
    
    positive_count = sum(1 for ev in eigenvalues if ev > tolerance)
    negative_count = sum(1 for ev in eigenvalues if ev < -tolerance)
    
    if positive_count == len(eigenvalues):
        return ('local_minimum', eigenvalues)
    elif negative_count == len(eigenvalues):
        return ('local_maximum', eigenvalues)
    else:
        return ('saddle_point', eigenvalues)


def get_critical_point_info(expr, point: Dict[str, float]) -> Dict:
    """
    Get full classification info for a critical point.
    
    Args:
        expr: SymPy expression or string
        point: Dictionary mapping variable names to values
        
    Returns:
        Dictionary with:
        - 'classification': Type of critical point
        - 'eigenvalues': List of Hessian eigenvalues
        - 'hessian': Numerical Hessian matrix
        - 'gradient': Gradient at the point (should be ~0 for critical point)
        - 'is_critical_point': Whether gradient is approximately zero
        
    Example:
        >>> info = get_critical_point_info("x^2 + y^2", {'x': 0, 'y': 0})
        >>> info['classification']
        'local_minimum'
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    # Compute gradient at point
    try:
        grad_values = gradient_at_point(expr, point)
    except ValueError as e:
        return {'error': str(e)}
    
    # Check if gradient is approximately zero (critical point)
    tolerance = 1e-8
    is_critical = all(abs(g) < tolerance for g in grad_values)
    
    # Compute Hessian at point
    try:
        H_numerical, variables = hessian_at_point(expr, point)
    except ValueError as e:
        return {'error': str(e)}
    
    # Classify
    classification, eigenvalues = classify_critical_point(H_numerical)
    
    return {
        'classification': classification,
        'eigenvalues': eigenvalues,
        'hessian': H_numerical,
        'gradient': grad_values,
        'is_critical_point': is_critical,
        'warning': None if is_critical else 'Gradient is not zero - may not be a critical point'
    }


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def format_gradient(grad: List, variables: List[Symbol]) -> str:
    """Format gradient as a readable string."""
    components = [f"∂f/∂{var} = {g}" for var, g in zip(variables, grad)]
    return "\n".join(components)


def format_hessian(H: Matrix, variables: List[Symbol]) -> str:
    """Format Hessian matrix as a readable string."""
    n = len(variables)
    
    # Header
    header = "    " + "  ".join(f"{var:>10}" for var in variables)
    
    lines = [header]
    for i, var in enumerate(variables):
        row_str = f"{var:>3} |"
        for j in range(n):
            row_str += f" {str(H[i,j]):>10}"
        lines.append(row_str)
    
    return "\n".join(lines)


def format_gradient_vector(grad: List, variables: List[Symbol]) -> str:
    """Format gradient as vector notation."""
    components = ", ".join(str(g) for g in grad)
    return f"[{components}]"

