"""
Gradus Week 1 - Single Variable Derivatives

This module provides:
- Expression parsing with edge case handling
- Single variable derivatives up to 3rd order
- Domain checking for function evaluation
"""

from .parsing import (
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
)

from .single_var import (
    derivative,
    first_derivative,
    second_derivative,
    third_derivative,
    nth_derivative,
    all_derivatives,
    derivative_at_point,
    is_constant_expression,
)

from .domain import (
    check_evaluation_point,
    get_domain_restrictions,
    validate_expression_size,
    check_abs_derivative_domain,
    DomainError,
    LargeNumberError,
    COMPUTATION_TIMEOUT,
)

