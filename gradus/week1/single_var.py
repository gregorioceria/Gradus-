"""
Single variable derivative computation module.

This module provides functions to compute derivatives of 
single-variable functions up to the third derivative.
"""

from sympy import diff, Symbol, simplify, Abs, sign, Piecewise
from typing import Union, List, Tuple, Optional
from .parsing import parse_expression, get_variables, create_symbol


def _make_variables_real(expr):
    """
    Replace variables with real-valued symbols for cleaner derivatives.
    Especially useful for abs(x) which gives sign(x) for real x.
    """
    from sympy import Symbol
    
    replacements = {}
    for var in expr.free_symbols:
        if not var.is_real:
            real_var = Symbol(str(var), real=True)
            replacements[var] = real_var
    
    if replacements:
        return expr.subs(replacements), replacements
    return expr, {}


def _restore_variables(expr, replacements):
    """Restore original variable names after computation."""
    if replacements:
        inverse = {v: k for k, v in replacements.items()}
        return expr.subs(inverse)
    return expr


class ConstantExpressionInfo:
    """Information about a constant expression."""
    def __init__(self, value, message: str):
        self.value = value
        self.message = message
        self.is_constant = True


def is_constant_expression(expr) -> bool:
    """Check if an expression is a constant (has no free variables)."""
    if isinstance(expr, str):
        expr = parse_expression(expr)
    return len(get_variables(expr)) == 0


def _get_single_variable(expr, var=None) -> Symbol:
    """
    Get the differentiation variable from an expression.
    
    If var is provided, use it. Otherwise, if the expression has
    exactly one variable, use that. Otherwise raise an error.
    
    Args:
        expr: SymPy expression
        var: Optional variable to differentiate with respect to
        
    Returns:
        Symbol to differentiate with respect to
        
    Raises:
        ValueError: If variable cannot be determined
    """
    if var is not None:
        if isinstance(var, str):
            return create_symbol(var)
        return var
    
    variables = get_variables(expr)
    
    if len(variables) == 0:
        raise ValueError("Expression has no variables to differentiate")
    elif len(variables) == 1:
        return variables.pop()
    else:
        raise ValueError(
            f"Expression has multiple variables {variables}. "
            "Please specify which variable to differentiate with respect to."
        )


def derivative(expr, var=None, order: int = 1, simplify_result: bool = True):
    """
    Compute the derivative of an expression.
    
    Args:
        expr: SymPy expression or string
        var: Variable to differentiate with respect to (optional for single-var)
        order: Order of derivative (1, 2, or 3)
        simplify_result: Whether to simplify the result
        
    Returns:
        SymPy expression representing the derivative
        
    Raises:
        ValueError: If order is not between 1 and 3
        
    Examples:
        >>> derivative("x^2", order=1)
        2*x
        >>> derivative("x^3", order=2)
        6*x
        >>> derivative("sin(x)", order=3)
        -cos(x)
    """
    if order < 1 or order > 3:
        raise ValueError("Order must be between 1 and 3")
    
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    # Make variables real for cleaner abs() derivatives
    expr_real, replacements = _make_variables_real(expr)
    
    var = _get_single_variable(expr, var)
    
    # Get corresponding real variable
    if replacements and var in replacements:
        var_real = replacements[var]
    else:
        var_real = var
    
    result = diff(expr_real, var_real, order)
    
    # Restore original variable names
    result = _restore_variables(result, replacements)
    
    if simplify_result:
        result = simplify(result)
    
    return result


def first_derivative(expr, var=None, simplify_result: bool = True):
    """
    Compute the first derivative (f').
    
    Args:
        expr: SymPy expression or string
        var: Variable to differentiate with respect to
        simplify_result: Whether to simplify the result
        
    Returns:
        First derivative as SymPy expression
        
    Examples:
        >>> first_derivative("x^3 + 2x^2 + x")
        3*x**2 + 4*x + 1
    """
    return derivative(expr, var, order=1, simplify_result=simplify_result)


def second_derivative(expr, var=None, simplify_result: bool = True):
    """
    Compute the second derivative (f'').
    
    Args:
        expr: SymPy expression or string
        var: Variable to differentiate with respect to
        simplify_result: Whether to simplify the result
        
    Returns:
        Second derivative as SymPy expression
        
    Examples:
        >>> second_derivative("x^3 + 2x^2 + x")
        6*x + 4
    """
    return derivative(expr, var, order=2, simplify_result=simplify_result)


def third_derivative(expr, var=None, simplify_result: bool = True):
    """
    Compute the third derivative (f''').
    
    Args:
        expr: SymPy expression or string
        var: Variable to differentiate with respect to
        simplify_result: Whether to simplify the result
        
    Returns:
        Third derivative as SymPy expression
        
    Examples:
        >>> third_derivative("x^3 + 2x^2 + x")
        6
    """
    return derivative(expr, var, order=3, simplify_result=simplify_result)


def nth_derivative(expr, var=None, n: int = 1, simplify_result: bool = True):
    """
    Compute the n-th derivative (generalized version, allows any positive n).
    
    Note: The main derivative() function is limited to orders 1-3 as per
    requirements. This function is provided for flexibility.
    
    Args:
        expr: SymPy expression or string
        var: Variable to differentiate with respect to
        n: Order of derivative (any positive integer)
        simplify_result: Whether to simplify the result
        
    Returns:
        n-th derivative as SymPy expression
    """
    if n < 1:
        raise ValueError("Derivative order must be at least 1")
    
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    var = _get_single_variable(expr, var)
    
    result = diff(expr, var, n)
    
    if simplify_result:
        result = simplify(result)
    
    return result


def all_derivatives(expr, var=None, simplify_result: bool = True) -> dict:
    """
    Compute all derivatives up to the third order.
    
    Args:
        expr: SymPy expression or string
        var: Variable to differentiate with respect to
        simplify_result: Whether to simplify the results
        
    Returns:
        Dictionary with keys 'f', 'f_prime', 'f_double_prime', 'f_triple_prime'
        Also includes 'is_constant' and 'message' if expression is a constant.
        
    Examples:
        >>> all_derivatives("x^4")
        {'f': x**4, 'f_prime': 4*x**3, 'f_double_prime': 12*x**2, 'f_triple_prime': 24*x}
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    # Check if expression is a constant
    if is_constant_expression(expr):
        return {
            'f': expr,
            'f_prime': 0,
            'f_double_prime': 0,
            'f_triple_prime': 0,
            'is_constant': True,
            'message': "This is a constant. Derivative of a constant is 0."
        }
    
    var = _get_single_variable(expr, var)
    
    return {
        'f': expr,
        'f_prime': derivative(expr, var, 1, simplify_result),
        'f_double_prime': derivative(expr, var, 2, simplify_result),
        'f_triple_prime': derivative(expr, var, 3, simplify_result),
        'is_constant': False
    }


def derivative_at_point(expr, var=None, point: float = 0, order: int = 1) -> float:
    """
    Evaluate the derivative at a specific point.
    
    Args:
        expr: SymPy expression or string
        var: Variable to differentiate with respect to
        point: Value at which to evaluate
        order: Order of derivative
        
    Returns:
        Numerical value of derivative at the point
        
    Examples:
        >>> derivative_at_point("x^2", point=3, order=1)
        6
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    var = _get_single_variable(expr, var)
    deriv = derivative(expr, var, order)
    
    return float(deriv.subs(var, point))


def show_differentiation_steps(expr, var=None) -> List[Tuple[str, str]]:
    """
    Show intermediate steps for the first derivative.
    
    This is an educational feature to help understand how 
    derivatives are computed step by step.
    
    Args:
        expr: SymPy expression or string
        var: Variable to differentiate with respect to
        
    Returns:
        List of (description, expression) tuples
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    var = _get_single_variable(expr, var)
    
    steps = []
    steps.append(("Original function f(x)", str(expr)))
    
    f_prime = diff(expr, var)
    steps.append(("First derivative f'(x)", str(f_prime)))
    
    f_prime_simplified = simplify(f_prime)
    if f_prime_simplified != f_prime:
        steps.append(("Simplified f'(x)", str(f_prime_simplified)))
    
    return steps

