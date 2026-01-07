"""
Parsing module for mathematical expressions.

This module provides utilities to parse string expressions into
SymPy symbolic expressions that can be manipulated mathematically.

Handles edge cases:
- Empty function arguments: f( )
- Unbalanced parentheses: (x+1
- Unsupported functions
- Division by zero: 1/0
- Malformed operators: x**2/ x, ++, --, ****, etc.
- Euler's number vs variable 'e'
- European decimal notation: 1.000,5 vs 1,000.5
"""

import re
from sympy import sympify, symbols, Symbol, E, pi, oo, zoo, nan
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
from typing import Union, Set, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# Standard transformations for parsing
TRANSFORMATIONS = (
    standard_transformations + 
    (implicit_multiplication_application, convert_xor)
)

# Supported functions in SymPy
SUPPORTED_FUNCTIONS = {
    # Trigonometric
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
    'arcsin', 'arccos', 'arctan',  # Aliases
    # Hyperbolic
    'sinh', 'cosh', 'tanh', 'coth',
    'asinh', 'acosh', 'atanh',
    # Exponential and logarithmic
    'exp', 'log', 'ln', 'log10', 'log2',
    # Roots and powers
    'sqrt', 'cbrt', 'root',
    # Other
    'abs', 'sign', 'floor', 'ceil', 'factorial',
    'gamma', 'erf',
}

# Regex patterns for validation
EMPTY_FUNCTION_PATTERN = re.compile(r'(\w+)\s*\(\s*\)')
CONSECUTIVE_OPERATORS_PATTERN = re.compile(r'[\+\-]{3,}|\*{3,}|/{2,}|\*\*\*+')
MALFORMED_OPERATOR_PATTERN = re.compile(r'(?<!\*)\*\s*/|/\s*\*(?!\*)|[\+\-]\s*[/]|[/]\s*[\+]')
DIVISION_BY_ZERO_PATTERN = re.compile(r'/\s*0(?![0-9\.])')
HANGING_OPERATOR_PATTERN = re.compile(r'[\+\-\*/\^]\s*$|^\s*[/\^]')


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ParsingError(ValueError):
    """Base exception for parsing errors."""
    pass


class EmptyExpressionError(ParsingError):
    """Raised when expression is empty."""
    pass


class UnbalancedParenthesesError(ParsingError):
    """Raised when parentheses are not balanced."""
    pass


class EmptyFunctionArgumentError(ParsingError):
    """Raised when a function has empty arguments like f()."""
    pass


class UnsupportedFunctionError(ParsingError):
    """Raised when an unsupported function is used."""
    pass


class MalformedOperatorError(ParsingError):
    """Raised when operators are malformed like ++, --, x**/, etc."""
    pass


class DivisionByZeroError(ParsingError):
    """Raised when division by zero is detected in the expression."""
    pass


class InvalidNumberFormatError(ParsingError):
    """Raised when number format is invalid."""
    pass


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def _check_empty_expression(expr_string: str) -> None:
    """Check if expression is empty or whitespace only."""
    if not expr_string or not expr_string.strip():
        raise EmptyExpressionError(
            "Expression cannot be empty. "
            "Please enter a valid mathematical expression."
        )


def _check_balanced_parentheses(expr_string: str) -> None:
    """
    Check if parentheses are properly balanced.
    
    Raises:
        UnbalancedParenthesesError: If parentheses don't match
    """
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for i, char in enumerate(expr_string):
        if char in pairs:
            stack.append((char, i))
        elif char in pairs.values():
            if not stack:
                raise UnbalancedParenthesesError(
                    f"Closing '{char}' at position {i} has no matching opening bracket. "
                    f"Expression: '{expr_string}'"
                )
            open_char, open_pos = stack.pop()
            if pairs[open_char] != char:
                raise UnbalancedParenthesesError(
                    f"Mismatched brackets: '{open_char}' at position {open_pos} "
                    f"closed with '{char}' at position {i}. "
                    f"Expression: '{expr_string}'"
                )
    
    if stack:
        open_char, open_pos = stack[0]
        raise UnbalancedParenthesesError(
            f"Opening '{open_char}' at position {open_pos} is never closed. "
            f"Expression: '{expr_string}'"
        )


def _check_empty_function_arguments(expr_string: str) -> None:
    """
    Check for functions with empty arguments like sin(), f(), etc.
    
    Raises:
        EmptyFunctionArgumentError: If empty function call found
    """
    matches = EMPTY_FUNCTION_PATTERN.findall(expr_string)
    
    for func_name in matches:
        if func_name.lower() in SUPPORTED_FUNCTIONS or func_name[0].isupper():
            raise EmptyFunctionArgumentError(
                f"Function '{func_name}()' requires an argument. "
                f"Example: {func_name}(x)"
            )


def _check_unsupported_functions(expr_string: str) -> None:
    """
    Check for functions that are not supported.
    
    Raises:
        UnsupportedFunctionError: If unsupported function found
    """
    func_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    matches = func_pattern.findall(expr_string)
    
    for func_name in matches:
        func_lower = func_name.lower()
        if (func_lower in SUPPORTED_FUNCTIONS or 
            len(func_name) == 1 or
            func_name in ['pi', 'Pi', 'PI', 'E', 'I']):
            continue
        
        similar = _find_similar_function(func_lower)
        if similar:
            raise UnsupportedFunctionError(
                f"Function '{func_name}' is not supported. "
                f"Did you mean '{similar}'? "
                f"Available functions: {', '.join(sorted(SUPPORTED_FUNCTIONS))}"
            )
        else:
            raise UnsupportedFunctionError(
                f"Function '{func_name}' is not recognized. "
                f"Supported functions: {', '.join(sorted(SUPPORTED_FUNCTIONS))}"
            )


def _find_similar_function(func_name: str) -> Optional[str]:
    """Find a similar supported function (for typo suggestions)."""
    for supported in SUPPORTED_FUNCTIONS:
        if len(func_name) == len(supported):
            diff = sum(1 for a, b in zip(func_name, supported) if a != b)
            if diff <= 2:
                return supported
        elif abs(len(func_name) - len(supported)) == 1:
            if func_name in supported or supported in func_name:
                return supported
    return None


def _check_malformed_operators(expr_string: str) -> None:
    """
    Check for malformed operator sequences.
    
    Examples of invalid: */, /*, ++-, x^/2, ***
    Note: ** is valid (power operator)
    
    Raises:
        MalformedOperatorError: If malformed operators found
    """
    cleaned = expr_string.replace('**', '@POW@')
    
    if CONSECUTIVE_OPERATORS_PATTERN.search(cleaned):
        raise MalformedOperatorError(
            f"Invalid consecutive operators in expression. "
            f"Check for sequences like '+++', '---', '***', '//'. "
            f"Expression: '{expr_string}'"
        )
    
    if MALFORMED_OPERATOR_PATTERN.search(cleaned):
        raise MalformedOperatorError(
            f"Invalid operator sequence (e.g., '*/', '/*'). "
            f"Expression: '{expr_string}'"
        )
    
    if HANGING_OPERATOR_PATTERN.search(expr_string):
        raise MalformedOperatorError(
            f"Expression cannot start with '/', '^' "
            f"or end with an operator. "
            f"Expression: '{expr_string}'"
        )


def _check_division_by_zero(expr_string: str) -> None:
    """
    Check for obvious division by zero in the expression.
    
    Raises:
        DivisionByZeroError: If /0 pattern found
    """
    if DIVISION_BY_ZERO_PATTERN.search(expr_string):
        raise DivisionByZeroError(
            f"Division by zero detected in expression. "
            f"Expression: '{expr_string}'"
        )


def _normalize_number_format(expr_string: str, european_format: bool = False) -> str:
    """
    Normalize number format handling decimal points and thousand separators.
    
    Args:
        expr_string: Expression string
        european_format: If True, treat comma as decimal separator
                        (e.g., 3,14 -> 3.14 and 1.000 -> 1000)
    
    Returns:
        Normalized expression string
    """
    if not european_format:
        result = re.sub(r'(\d),(\d{3})(?!\d)', r'\1\2', expr_string)
        return result
    
    def convert_european(match):
        num = match.group(0)
        num = num.replace('.', '')
        num = num.replace(',', '.')
        return num
    
    european_num_pattern = re.compile(r'\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+,\d+')
    result = european_num_pattern.sub(convert_european, expr_string)
    
    return result


def _convert_absolute_value(expr_string: str) -> str:
    """
    Convert |x| notation to abs(x).
    
    Handles nested expressions like |x+1| -> abs(x+1)
    """
    result = []
    i = 0
    while i < len(expr_string):
        if expr_string[i] == '|':
            # Find matching |
            depth = 0
            j = i + 1
            while j < len(expr_string):
                if expr_string[j] == '|' and depth == 0:
                    # Found matching |
                    inner = expr_string[i+1:j]
                    result.append(f'abs({inner})')
                    i = j + 1
                    break
                elif expr_string[j] == '(':
                    depth += 1
                elif expr_string[j] == ')':
                    depth -= 1
                j += 1
            else:
                # No matching | found, keep as is
                result.append(expr_string[i])
                i += 1
        else:
            result.append(expr_string[i])
            i += 1
    return ''.join(result)


def _preprocess_expression(expr_string: str, euler_as_constant: bool = True) -> str:
    """
    Preprocess the expression string before parsing.
    
    Handles:
    - Absolute value: |x| -> abs(x)
    - Power notation: ^ -> **
    - Natural log: ln -> log
    - Euler's number: e -> E (if euler_as_constant)
    - Whitespace cleanup
    
    Args:
        expr_string: Raw expression string
        euler_as_constant: If True, 'e' alone is treated as Euler's number
        
    Returns:
        Preprocessed expression string
    """
    result = expr_string.strip()
    
    # Convert |x| to abs(x)
    result = _convert_absolute_value(result)
    
    result = result.replace('^', '**')
    result = re.sub(r'\bln\b', 'log', result)
    result = re.sub(r'\barcsin\b', 'asin', result)
    result = re.sub(r'\barccos\b', 'acos', result)
    result = re.sub(r'\barctan\b', 'atan', result)
    
    if euler_as_constant:
        result = re.sub(r'(?<![a-zA-Z])e(?![a-zA-Z])', 'E', result)
    
    return result


def validate_expression_syntax(expr_string: str, 
                                european_format: bool = False,
                                euler_as_constant: bool = True) -> str:
    """
    Validate expression syntax and return preprocessed string.
    
    Runs all validation checks and preprocessing steps.
    
    Args:
        expr_string: Raw expression string
        european_format: Use European number format
        euler_as_constant: Treat 'e' as Euler's number
        
    Returns:
        Validated and preprocessed expression string
        
    Raises:
        ParsingError: If any validation fails
    """
    _check_empty_expression(expr_string)
    expr_string = _normalize_number_format(expr_string, european_format)
    _check_balanced_parentheses(expr_string)
    _check_empty_function_arguments(expr_string)
    _check_malformed_operators(expr_string)
    _check_division_by_zero(expr_string)
    expr_string = _preprocess_expression(expr_string, euler_as_constant)
    _check_unsupported_functions(expr_string)
    
    return expr_string


# =============================================================================
# MAIN PARSING FUNCTIONS
# =============================================================================

def parse_expression(expr_string: str, 
                     local_dict: dict = None,
                     european_format: bool = False,
                     euler_as_constant: bool = True,
                     strict_validation: bool = True):
    """
    Parse a string expression into a SymPy expression.
    
    Supports:
    - Standard mathematical operations: +, -, *, /, **
    - Implicit multiplication: 2x -> 2*x, x(x+1) -> x*(x+1)
    - Common functions: sin, cos, tan, exp, log, ln, sqrt
    - Constants: pi, E (Euler's number), I (imaginary unit)
    
    Args:
        expr_string: Mathematical expression as string
        local_dict: Optional dictionary of local variables/symbols
        european_format: If True, use European number format (comma as decimal)
        euler_as_constant: If True, 'e' is Euler's number; if False, 'e' is a variable
        strict_validation: If True, run all validation checks
        
    Returns:
        SymPy expression object
        
    Raises:
        ParsingError: If expression has syntax errors
        EmptyExpressionError: If expression is empty
        UnbalancedParenthesesError: If parentheses don't match
        EmptyFunctionArgumentError: If function has no argument
        UnsupportedFunctionError: If function is not supported
        MalformedOperatorError: If operators are malformed
        DivisionByZeroError: If dividing by zero
        
    Examples:
        >>> parse_expression("x^2 + 2x + 1")
        x**2 + 2*x + 1
        >>> parse_expression("sin(x) + cos(x)")
        sin(x) + cos(x)
        >>> parse_expression("3,14 * x", european_format=True)
        3.14*x
    """
    if strict_validation:
        expr_string = validate_expression_syntax(
            expr_string, 
            european_format, 
            euler_as_constant
        )
    else:
        _check_empty_expression(expr_string)
        expr_string = _normalize_number_format(expr_string, european_format)
        expr_string = _preprocess_expression(expr_string, euler_as_constant)
    
    try:
        expr = parse_expr(
            expr_string,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS
        )
        
        if expr.has(oo, -oo, zoo, nan):
            raise DivisionByZeroError(
                f"Expression results in infinite or undefined value. "
                f"Check for division by zero."
            )
        
        return expr
        
    except ParsingError:
        raise
    except SyntaxError as e:
        raise ParsingError(
            f"Syntax error in expression: {e}. "
            f"Please check the expression syntax."
        )
    except Exception as e:
        raise ParsingError(
            f"Cannot parse expression '{expr_string}': {e}"
        )


def get_variables(expr) -> Set[Symbol]:
    """
    Extract all free variables from an expression.
    
    Args:
        expr: SymPy expression or string
        
    Returns:
        Set of Symbol objects representing variables
        
    Examples:
        >>> get_variables("x^2 + y")
        {x, y}
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    return expr.free_symbols


def create_symbol(name: str) -> Symbol:
    """
    Create a SymPy symbol with the given name.
    
    Args:
        name: Name of the symbol
        
    Returns:
        SymPy Symbol object
    """
    return Symbol(name)


def create_symbols(*names: str):
    """
    Create multiple SymPy symbols.
    
    Args:
        *names: Variable number of symbol names
        
    Returns:
        Tuple of Symbol objects
        
    Examples:
        >>> x, y, z = create_symbols('x', 'y', 'z')
    """
    return symbols(' '.join(names))


def validate_expression(expr_string: str, 
                        european_format: bool = False,
                        euler_as_constant: bool = True) -> tuple:
    """
    Check if a string is a valid mathematical expression.
    
    Args:
        expr_string: Expression to validate
        european_format: Use European number format
        euler_as_constant: Treat 'e' as Euler's number
        
    Returns:
        Tuple of (is_valid: bool, error_message: str or None)
    """
    try:
        parse_expression(
            expr_string, 
            european_format=european_format,
            euler_as_constant=euler_as_constant
        )
        return (True, None)
    except ParsingError as e:
        return (False, str(e))
    except Exception as e:
        return (False, f"Unexpected error: {e}")


def simplify_expression(expr):
    """
    Simplify a mathematical expression.
    
    Args:
        expr: SymPy expression or string
        
    Returns:
        Simplified SymPy expression
    """
    from sympy import simplify
    
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    return simplify(expr)


def expression_to_string(expr, use_unicode: bool = True) -> str:
    """
    Convert a SymPy expression to a readable string.
    
    Args:
        expr: SymPy expression
        use_unicode: Whether to use Unicode characters
        
    Returns:
        String representation of the expression
    """
    from sympy import pretty
    
    if use_unicode:
        return pretty(expr, use_unicode=True)
    return str(expr)


def expression_to_latex(expr) -> str:
    """
    Convert a SymPy expression to LaTeX format.
    
    Args:
        expr: SymPy expression
        
    Returns:
        LaTeX string representation
    """
    from sympy import latex
    
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    return latex(expr)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_supported_functions() -> Set[str]:
    """Return the set of supported function names."""
    return SUPPORTED_FUNCTIONS.copy()


def format_error_message(error: ParsingError) -> str:
    """
    Format an error message for user display.
    
    Args:
        error: ParsingError instance
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    type_translations = {
        'EmptyExpressionError': '[ERROR] Empty Expression',
        'UnbalancedParenthesesError': '[ERROR] Unbalanced Parentheses',
        'EmptyFunctionArgumentError': '[ERROR] Missing Function Argument',
        'UnsupportedFunctionError': '[ERROR] Unsupported Function',
        'MalformedOperatorError': '[ERROR] Malformed Operators',
        'DivisionByZeroError': '[ERROR] Division by Zero',
        'InvalidNumberFormatError': '[ERROR] Invalid Number Format',
        'ParsingError': '[ERROR] Parsing Error',
    }
    
    title = type_translations.get(error_type, '[ERROR]')
    return f"{title}\n{str(error)}"

