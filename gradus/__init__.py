"""
Gradus - Symbolic Derivative Calculator

A symbolic mathematics tool for computing derivatives of single and 
multivariable functions, including gradients and Hessian matrices.

Structure:
- week1: Parsing + Single variable derivatives (up to 3rd order)
- week2: Multivariable derivatives, gradient, Hessian
- week3: 3D plotting, Streamlit UI (coming soon)
- week4: Refinements and report (coming soon)
"""

__version__ = "0.2.0"
__author__ = "Your Name"

# Week 1 exports
from .week1 import (
    # Parsing
    parse_expression,
    get_variables,
    validate_expression,
    expression_to_latex,
    expression_to_string,
    # Exceptions
    ParsingError,
    EmptyExpressionError,
    UnbalancedParenthesesError,
    EmptyFunctionArgumentError,
    UnsupportedFunctionError,
    MalformedOperatorError,
    DivisionByZeroError,
    InvalidNumberFormatError,
    # Derivatives
    derivative,
    first_derivative,
    second_derivative,
    third_derivative,
    nth_derivative,
    all_derivatives,
    derivative_at_point,
)

# Week 2 exports
from .week2 import (
    partial_derivative,
    all_partial_derivatives,
    gradient,
    hessian,
    hessian_at_point,
    gradient_at_point,
    mixed_partial,
)
