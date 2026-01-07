#!/usr/bin/env python3
"""
Gradus CLI - Command Line Interface for symbolic derivative calculator.

This is a simple CLI to test the derivative computation logic.
Run with: python cli.py

Supports:
- Week 1: Single variable derivatives (up to 3rd order)
- Week 2: Multivariable derivatives (gradient, Hessian)
- Week 3: Function visualization (1D, 2D, 3D plots)
"""

import sys
import re
import signal
from sympy import pi, E, Rational, N, sympify, Abs, Symbol

from gradus import (
    parse_expression,
    get_variables,
    first_derivative,
    second_derivative,
    third_derivative,
    ParsingError,
)
from gradus.week1 import (
    all_derivatives, 
    derivative_at_point,
    check_evaluation_point,
    get_domain_restrictions,
    validate_expression_size,
    check_abs_derivative_domain,
    COMPUTATION_TIMEOUT,
)
from gradus.week1.parsing import expression_to_latex, format_error_message

# Week 2 imports
from gradus.week2 import (
    partial_derivative,
    all_partial_derivatives,
    gradient,
    hessian,
    gradient_at_point,
    hessian_at_point,
)
from gradus.week2.multi_var import (
    TooManyVariablesError,
    NoVariablesError,
    MAX_VARIABLES,
    format_gradient_vector,
)

# Week 3 imports - plotting (optional, may not be installed)
PLOTTING_AVAILABLE = False
try:
    from gradus.week3 import (
        plot_1d,
        plot_2d_heatmap,
        plot_2d_gradient_field,
        plot_3d_surface,
        plot_function,
    )
    from gradus.week3.plotting import show_plot
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    pass  # matplotlib not installed


# =============================================================================
# TIMEOUT HANDLING (Unix only, graceful fallback on Windows)
# =============================================================================

class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Computation timed out")


def with_timeout(func, timeout_seconds=COMPUTATION_TIMEOUT):
    """Execute a function with timeout. Falls back gracefully on Windows."""
    try:
        # Unix-like systems
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            result = func()
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            raise
    except AttributeError:
        # Windows doesn't support SIGALRM, just run without timeout
        return func()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_point_value(value_str: str):
    """
    Parse a point value that can be:
    - A number: "3", "3.14", "-2.5"
    - A fraction: "1/2", "3/4", "-1/3"
    - A constant: "pi", "e"
    - An expression: "pi/2", "e/3", "2*pi", "sqrt(2)"
    - Powers: "2^(1/3)", "8^(1/2)", "27^(1/3)"
    
    Returns:
        float value
    """
    value_str = value_str.strip().lower()
    
    # Replace ^ with ** for power notation
    value_str = value_str.replace('^', '**')
    
    # Replace 'e' with Euler's number (but not in 'exp')
    value_str = re.sub(r'\be\b', 'E', value_str)
    
    try:
        expr = sympify(value_str)
        
        # Check that the expression has no free variables (must be a number)
        if expr.free_symbols:
            vars_str = ', '.join(str(v) for v in expr.free_symbols)
            raise ValueError(f"Expression contains variables ({vars_str}). Enter a numeric value")
        
        # Try to evaluate
        evaluated = N(expr)
        
        # Check for complex results
        if hasattr(evaluated, 'is_real') and not evaluated.is_real:
            # Special message for odd roots of negative numbers
            if '**' in value_str and '-' in value_str:
                raise ValueError(f"Result is complex. For real cube roots of negative numbers, use cbrt(-8) instead of (-8)^(1/3)")
            raise ValueError(f"Result is not a real number (complex value)")
        
        result = float(evaluated)
        
        # NaN check
        if result != result:
            raise ValueError(f"Result is undefined (NaN)")
            
        return result
    except ValueError:
        raise
    except TypeError as e:
        if 'complex' in str(e).lower():
            raise ValueError(f"Result is complex, not a real number")
        raise ValueError(f"Cannot parse '{value_str}' as a number")
    except Exception as e:
        raise ValueError(f"Cannot parse '{value_str}' as a number")


def parse_multivar_point(prompt_vars: list) -> dict:
    """
    Parse a multivariate point from user input.
    
    Args:
        prompt_vars: List of variable symbols
        
    Returns:
        Dictionary mapping variable names to values
    """
    var_names = [str(v) for v in prompt_vars]
    print(f"  Enter values for: {', '.join(var_names)}")
    print(f"  Format: {var_names[0]}=1, {var_names[1]}=2, ... (or 'n' to skip)")
    
    user_input = input("  Point: ").strip()
    
    if user_input.lower() in ['n', 'no', '']:
        return None
    
    # Parse format: x=1, y=2, z=3 or x=1 y=2 z=3
    point = {}
    
    # Split by comma or space
    parts = re.split(r'[,\s]+', user_input)
    
    for part in parts:
        if '=' not in part:
            raise ValueError(f"Invalid format '{part}'. Use var=value (e.g., x=1)")
        
        var_part, val_part = part.split('=', 1)
        var_name = var_part.strip()
        
        if var_name not in var_names:
            raise ValueError(f"Unknown variable '{var_name}'. Expected: {', '.join(var_names)}")
        
        try:
            value = parse_point_value(val_part.strip())
            point[var_name] = value
        except ValueError as e:
            raise ValueError(f"Invalid value for {var_name}: {e}")
    
    # Check all variables are provided
    missing = set(var_names) - set(point.keys())
    if missing:
        raise ValueError(f"Missing values for: {', '.join(missing)}")
    
    return point


# =============================================================================
# HEADER AND HELP
# =============================================================================

def print_header():
    """Print the CLI header."""
    print("=" * 65)
    print("  GRADUS - Symbolic Derivative Calculator")
    print("  Supports: Single variable (1 var) and Multivariable (2-6 vars)")
    if PLOTTING_AVAILABLE:
        print("  Plotting: Available (matplotlib installed)")
    else:
        print("  Plotting: Not available (install matplotlib)")
    print("=" * 65)
    print()


def print_help():
    """Print help information."""
    plot_help = """
Plotting (if available):
  After entering function, you'll be asked if you want a graph.
  1D functions: Shows f(x), f'(x), and critical points
  2D functions: Shows 3D surface (can rotate with mouse)
""" if PLOTTING_AVAILABLE else ""

    print(f"""
Available commands:
  help          - Show this help message
  examples      - Show example expressions
  quit / exit   - Exit the program

Supported operations:
  +, -, *, /, ^ (or **)  - Basic arithmetic
  |x|, abs(x)            - Absolute value
  sin, cos, tan          - Trigonometric functions
  exp, log (or ln)       - Exponential and logarithm
  sqrt                   - Square root
  pi, e                  - Constants
  
Single Variable (1 var):
  f(x) = x^2 + 2x        -> Computes f', f'', f'''
  
Multivariable (2-6 vars):
  f(x,y) = x^2 + y^2     -> Computes gradient and Hessian
  
Evaluation points:
  Single var:  3, 1/2, pi, sqrt(2)
  Multi var:   x=1, y=2 (or x=1,y=2)
  
Examples:
  x^2 + 2x + 1           (single variable)
  x^2 + y^2 + xy         (two variables)
  x^2 + y^2 + z^2        (three variables)
{plot_help}""")


def print_examples():
    """Show example computations."""
    print("\n" + "-" * 55)
    print("SINGLE VARIABLE EXAMPLES")
    print("-" * 55)
    
    single_examples = [
        "x^2",
        "sin(x)",
        "exp(x) * cos(x)",
    ]
    
    for expr_str in single_examples:
        print(f"\nf(x) = {expr_str}")
        try:
            derivs = all_derivatives(expr_str)
            print(f"  f'(x)   = {derivs['f_prime']}")
            print(f"  f''(x)  = {derivs['f_double_prime']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "-" * 55)
    print("MULTIVARIABLE EXAMPLES")
    print("-" * 55)
    
    multi_examples = [
        "x^2 + y^2",
        "x^2 + y^2 + xy",
        "x*y*z",
    ]
    
    for expr_str in multi_examples:
        print(f"\nf = {expr_str}")
        try:
            grad, vars_list = gradient(expr_str)
            H, _ = hessian(expr_str)
            
            print(f"  Gradient: {format_gradient_vector(grad, vars_list)}")
            print(f"  Hessian:")
            n = len(vars_list)
            for i in range(n):
                row = [str(H[i,j]) for j in range(n)]
                print(f"    [{', '.join(row)}]")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("-" * 55 + "\n")


# =============================================================================
# SINGLE VARIABLE COMPUTATION (Week 1)
# =============================================================================

def compute_single_var(expr, expr_string: str, variables):
    """Compute and display single variable derivatives."""
    var = list(variables)[0]
    
    # Show domain restrictions if any
    restrictions = get_domain_restrictions(expr)
    if restrictions:
        print(f"\n  Domain restrictions:")
        for r in restrictions:
            print(f"    - {r}")
    
    print(f"{'-' * 55}")
    
    # Compute derivatives with timeout
    def compute():
        return all_derivatives(expr)
    
    try:
        derivs = with_timeout(compute, COMPUTATION_TIMEOUT)
    except TimeoutError:
        print(f"\n  [ERROR] Computation timed out after {COMPUTATION_TIMEOUT} seconds.")
        print("  The expression may be too complex.")
        return
    
    # Check if constant
    if derivs.get('is_constant'):
        print(f"\n  [INFO] {derivs['message']}")
    
    print(f"\n  f(x)    = {derivs['f']}")
    print(f"  f'(x)   = {derivs['f_prime']}")
    print(f"  f''(x)  = {derivs['f_double_prime']}")
    print(f"  f'''(x) = {derivs['f_triple_prime']}")
    
    # Skip LaTeX for constants
    if not derivs.get('is_constant'):
        print(f"\n  LaTeX for f'(x): {expression_to_latex(derivs['f_prime'])}")
    
    # Don't ask for evaluation if constant
    if derivs.get('is_constant'):
        print()
        return
    
    # Evaluate at a point?
    print()
    
    while True:
        eval_input = input("  Evaluate at a point? (number, fraction, pi, e, or 'n' to skip): ").strip()
        if eval_input.lower() in ['n', 'no', '']:
            break
        try:
            point = parse_point_value(eval_input)
            point_display = eval_input if eval_input.lower() in ['pi', 'e', 'pi/2', 'pi/4'] else f"{point:.6g}"
            
            # Check domain before evaluation
            is_valid, domain_error = check_evaluation_point(expr, var, point)
            if not is_valid:
                print(f"\n  [ERROR] {domain_error}")
                continue
            
            # Check for derivative domain issues (abs at 0)
            f_prime = derivs['f_prime']
            if f_prime.has(Abs) or (hasattr(f_prime, 'func') and 'sign' in str(f_prime)):
                valid, warning = check_abs_derivative_domain(point)
                if not valid:
                    print(f"\n  [WARNING] {warning}")
            
            print(f"\n  At x = {point_display} ({point:.6f}):")
            
            try:
                f_val = float(expr.subs(var, point))
                print(f"    f(x)    = {f_val:.6f}")
            except:
                print(f"    f(x)    = undefined")
            
            try:
                f1_val = derivative_at_point(expr, var=var, point=point, order=1)
                print(f"    f'(x)   = {f1_val:.6f}")
            except:
                print(f"    f'(x)   = undefined")
            
            try:
                f2_val = derivative_at_point(expr, var=var, point=point, order=2)
                print(f"    f''(x)  = {f2_val:.6f}")
            except:
                print(f"    f''(x)  = undefined")
            
            try:
                f3_val = derivative_at_point(expr, var=var, point=point, order=3)
                print(f"    f'''(x) = {f3_val:.6f}")
            except:
                print(f"    f'''(x) = undefined")
            
            print()
            
        except ValueError as ve:
            print(f"  {ve}. Try again or enter 'n'.")
        except Exception as e:
            print(f"  Cannot evaluate: {e}")
    
    print()


# =============================================================================
# MULTIVARIABLE COMPUTATION (Week 2)
# =============================================================================

def compute_multivar(expr, expr_string: str, variables):
    """Compute and display multivariable derivatives (gradient, Hessian)."""
    vars_list = sorted(list(variables), key=lambda x: str(x))
    n = len(vars_list)
    
    print(f"{'-' * 55}")
    
    # Compute gradient
    print(f"\n  Partial Derivatives:")
    partials = all_partial_derivatives(expr)
    for var in vars_list:
        pd = partials[var]
        print(f"    df/d{var} = {pd}")
    
    # Compute gradient
    grad, _ = gradient(expr)
    print(f"\n  Gradient:")
    print(f"    grad(f) = {format_gradient_vector(grad, vars_list)}")
    
    # Compute Hessian
    H, _ = hessian(expr)
    print(f"\n  Hessian Matrix:")
    
    # Header row
    header = "         " + "".join(f"{str(v):>12}" for v in vars_list)
    print(header)
    
    # Matrix rows
    for i, var_i in enumerate(vars_list):
        row_str = f"    {str(var_i):>4} |"
        for j in range(n):
            row_str += f"{str(H[i,j]):>12}"
        print(row_str)
    
    # LaTeX for Hessian
    print(f"\n  LaTeX for Hessian: {expression_to_latex(H)}")
    
    # Evaluate at a point?
    print()
    
    while True:
        try:
            point = parse_multivar_point(vars_list)
            if point is None:
                break
            
            # Display point
            point_str = ", ".join(f"{k}={v}" for k, v in point.items())
            print(f"\n  At ({point_str}):")
            
            # Evaluate function
            subs_dict = {Symbol(k): v for k, v in point.items()}
            try:
                f_val = float(N(expr.subs(subs_dict)))
                print(f"    f = {f_val:.6f}")
            except:
                print(f"    f = undefined")
            
            # Evaluate gradient
            try:
                grad_vals = gradient_at_point(expr, point)
                grad_str = "[" + ", ".join(f"{v:.6f}" for v in grad_vals) + "]"
                print(f"    grad(f) = {grad_str}")
            except Exception as e:
                print(f"    grad(f) = undefined ({e})")
            
            # Evaluate Hessian
            try:
                H_vals, _ = hessian_at_point(expr, point)
                print(f"    Hessian:")
                for row in H_vals:
                    row_str = "[" + ", ".join(f"{v:>10.4f}" for v in row) + "]"
                    print(f"      {row_str}")
            except Exception as e:
                print(f"    Hessian = undefined ({e})")
            
            print()
            
        except ValueError as ve:
            print(f"  [ERROR] {ve}")
        except Exception as e:
            print(f"  Cannot evaluate: {e}")
    
    print()


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def ask_for_plot() -> bool:
    """Ask user if they want to visualize the graph."""
    if not PLOTTING_AVAILABLE:
        return False
    
    response = input("  Visualize graph? (y/n): ").strip().lower()
    return response in ['y', 'yes', 's', 'si']


def show_plot_1d(expr, expr_string: str):
    """Show 1D plot for single variable function."""
    if not PLOTTING_AVAILABLE:
        return
    
    print("\n  [PLOTTING] Generating 1D graph...")
    try:
        fig = plot_1d(
            expr,
            show_derivative=True,
            show_second_derivative=False,
            show_critical_points=True,
            show_inflection_points=False,
        )
        plt.show()
    except Exception as e:
        print(f"  [ERROR] Could not generate plot: {e}")


def show_plot_2d(expr, expr_string: str):
    """Show plot options for 2-variable function."""
    if not PLOTTING_AVAILABLE:
        return
    
    print("\n  Plot types available:")
    print("    1. 3D Surface (default)")
    print("    2. Heatmap")
    print("    3. Gradient field")
    
    choice = input("  Choose (1-3, or Enter for default): ").strip()
    
    print("\n  [PLOTTING] Generating graph...")
    try:
        if choice == '2':
            fig = plot_2d_heatmap(expr)
        elif choice == '3':
            fig = plot_2d_gradient_field(expr)
        else:
            fig = plot_3d_surface(expr)
        
        plt.show()
    except Exception as e:
        print(f"  [ERROR] Could not generate plot: {e}")


# =============================================================================
# MAIN COMPUTATION ROUTER
# =============================================================================

def compute_derivatives(expr_string: str, want_plot: bool = False):
    """Compute and display derivatives based on number of variables."""
    try:
        # Parse and validate
        expr = parse_expression(expr_string)
        variables = get_variables(expr)
        num_vars = len(variables)
        
        # Check for large numbers
        is_valid, size_messages = validate_expression_size(expr)
        for msg in size_messages:
            if "Warning" in msg:
                print(f"\n  [WARNING] {msg}")
            else:
                print(f"\n  [ERROR] {msg}")
        
        if not is_valid:
            return
        
        # Display header
        print(f"\n{'-' * 55}")
        print(f"  Input: {expr_string}")
        print(f"  Parsed: {expr}")
        
        if num_vars == 0:
            print(f"  Variables: (none - this is a constant)")
            print(f"  Mode: Constant")
        elif num_vars == 1:
            print(f"  Variables: {list(variables)[0]}")
            print(f"  Mode: Single Variable")
        else:
            vars_str = ", ".join(str(v) for v in sorted(variables, key=str))
            print(f"  Variables: {vars_str}")
            print(f"  Mode: Multivariable ({num_vars} variables)")
        
        # Check max variables
        if num_vars > MAX_VARIABLES:
            print(f"\n  [ERROR] Too many variables ({num_vars}). Maximum is {MAX_VARIABLES}.")
            return
        
        # Check if plotting is possible (max 2 variables)
        if want_plot and num_vars > 2:
            print(f"\n  [INFO] Plotting not available for {num_vars} variables (max 2).")
            want_plot = False
        
        # Route to appropriate computation
        if num_vars <= 1:
            # Single variable or constant (Week 1)
            compute_single_var(expr, expr_string, variables)
            
            # Show plot if requested
            if want_plot and num_vars == 1:
                show_plot_1d(expr, expr_string)
        else:
            # Multivariable (Week 2)
            compute_multivar(expr, expr_string, variables)
            
            # Show plot if requested (only for 2 variables)
            if want_plot and num_vars == 2:
                show_plot_2d(expr, expr_string)
        
    except ParsingError as e:
        print(f"\n  [ERROR] {e}\n")
    except (TooManyVariablesError, NoVariablesError) as e:
        print(f"\n  [ERROR] {e}\n")
    except ValueError as e:
        print(f"\n  [ERROR] {e}\n")
    except Exception as e:
        print(f"\n  [ERROR] Unexpected error: {e}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main CLI loop."""
    print_header()
    print("Enter a function to differentiate (or 'help' for commands)")
    print()
    
    while True:
        try:
            user_input = input("f(x) = ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            cmd = user_input.lower()
            if cmd in ['quit', 'exit', 'q']:
                print("\nGoodbye!\n")
                break
            elif cmd == 'help':
                print_help()
                continue
            elif cmd == 'examples':
                print_examples()
                continue
            
            # Ask for plot before computing
            want_plot = ask_for_plot()
            
            # Treat as expression
            compute_derivatives(user_input, want_plot)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except EOFError:
            print("\nGoodbye!\n")
            break


if __name__ == "__main__":
    main()
