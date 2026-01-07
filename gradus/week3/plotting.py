"""
Plotting module for Gradus.

Provides visualization functions for single and multivariable functions.
- Clean plots without markers by default
- Better domain handling with visual indicators
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Optional, Callable, Dict
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Symbol, lambdify, solve, N, oo, zoo, nan, I, diff, latex
from week1.parsing import parse_expression, get_variables
from week1.single_var import first_derivative, second_derivative


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_RANGE = (-10, 10)
DEFAULT_POINTS = 2000  # Higher for better resolution near asymptotes
DEFAULT_POINTS_2D = 100

# Plot styling
STYLE_CONFIG = {
    'figure_size_1d': (12, 7),
    'figure_size_2d': (11, 9),
    'figure_size_3d': (13, 10),
    'function_color': '#0077B6',
    'derivative_color': '#E63946',
    'grid_alpha': 0.3,
    'line_width': 2.5,
    'domain_warning_color': '#FFE066',  # Yellow for domain warnings
}


# =============================================================================
# DOMAIN HANDLING
# =============================================================================

def _analyze_domain(expr) -> Dict:
    """
    Analyze expression to detect domain restrictions.
    Returns dict with domain info and recommended ranges.
    """
    expr_str = str(expr).lower()
    
    restrictions = []
    x_min_adjust = None
    y_min_adjust = None
    has_singularity = False
    
    # Log function - needs positive argument
    if 'log' in expr_str or 'ln(' in expr_str:
        restrictions.append("log requires positive argument")
        x_min_adjust = 0.01
    
    # Square root - needs non-negative
    if 'sqrt' in expr_str:
        restrictions.append("sqrt requires non-negative argument")
        x_min_adjust = 0 if x_min_adjust is None else max(0, x_min_adjust)
    
    # Division - potential singularity
    if '/x' in expr_str or '/ x' in expr_str or '1/x' in expr_str:
        restrictions.append("division by zero at x=0")
        has_singularity = True
    
    # Tangent - vertical asymptotes
    if 'tan(' in expr_str:
        restrictions.append("tan has vertical asymptotes at x = pi/2 + n*pi")
        has_singularity = True
    
    # arcsin/arccos - domain [-1, 1]
    if 'asin' in expr_str or 'acos' in expr_str:
        restrictions.append("asin/acos requires argument in [-1, 1]")
    
    return {
        'restrictions': restrictions,
        'x_min_adjust': x_min_adjust,
        'y_min_adjust': y_min_adjust,
        'has_singularity': has_singularity,
        'has_restrictions': len(restrictions) > 0
    }


def _get_safe_range(expr, var, default_range: Tuple[float, float] = DEFAULT_RANGE) -> Tuple[float, float]:
    """Determine a safe plotting range based on expression's domain."""
    domain_info = _analyze_domain(expr)
    x_min, x_max = default_range
    
    if domain_info['x_min_adjust'] is not None:
        x_min = max(x_min, domain_info['x_min_adjust'])
    
    # Ensure we have a valid range
    if x_min >= x_max:
        x_min = 0.01
        x_max = 10
    
    return (x_min, x_max)


def _create_lambda(expr, variables: List[Symbol]) -> Callable:
    """Create a numpy-compatible lambda function from a SymPy expression."""
    numpy_modules = ['numpy', {'Abs': np.abs}]
    return lambdify(variables, expr, modules=numpy_modules)


def _safe_evaluate(func: Callable, *args, max_value: float = 1e6) -> np.ndarray:
    """Safely evaluate a function, handling infinities and complex results."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = func(*args)
        
        # Get expected shape from input args
        expected_shape = args[0].shape if hasattr(args[0], 'shape') else None
        
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        
        # Broadcast scalar/constant results to match input shape
        if expected_shape is not None and result.shape != expected_shape:
            result = np.broadcast_to(result, expected_shape).astype(float)
        
        if np.iscomplexobj(result):
            mask = np.abs(result.imag) < 1e-10
            result = np.where(mask, result.real, np.nan)
        
        result = np.where(np.isinf(result), np.nan, result)
        result = np.where(np.abs(result) > max_value, np.nan, result)
        
        return result


def _break_at_discontinuities(y: np.ndarray) -> np.ndarray:
    """
    Insert NaN at discontinuities to prevent matplotlib from drawing 
    vertical lines across asymptotes (like in tan(x), 1/x, etc.)
    
    Strategy: Break line where there's a sign change with large magnitude on both sides.
    """
    y = y.copy()
    
    for i in range(len(y) - 1):
        # Skip if either value is already NaN
        if np.isnan(y[i]) or np.isnan(y[i + 1]):
            continue
        
        # Detect sign change with significant magnitude on both sides
        # This catches asymptotes where function goes from +large to -large
        if y[i] * y[i + 1] < 0:  # Sign change
            if abs(y[i]) > 3 and abs(y[i + 1]) > 3:
                y[i + 1] = np.nan
    
    return y


def _get_smart_ylim(y: np.ndarray, y_prime: np.ndarray = None, margin: float = 0.1) -> Tuple[float, float]:
    """
    Calculate smart y-axis limits based on data, excluding extreme outliers.
    Uses percentile-based approach to handle functions with asymptotes like tan(x).
    """
    # Combine all y data
    all_data = [y]
    if y_prime is not None:
        all_data.append(y_prime)
    
    combined = np.concatenate([d.flatten() for d in all_data])
    valid = combined[~np.isnan(combined)]
    
    if len(valid) == 0:
        return (-10, 10)
    
    # Use percentiles to exclude extreme outliers (useful for tan, 1/x, etc.)
    y_min = np.percentile(valid, 2)
    y_max = np.percentile(valid, 98)
    
    # Add margin
    y_range = y_max - y_min
    if y_range < 1e-10:
        y_range = 1
    
    y_min -= margin * y_range
    y_max += margin * y_range
    
    # Ensure reasonable limits
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    
    return (y_min, y_max)


def _to_latex(expr) -> str:
    """Convert SymPy expression to LaTeX string."""
    try:
        return f"${latex(expr)}$"
    except:
        return str(expr)


# =============================================================================
# 1D PLOTTING
# =============================================================================

def plot_1d(
    expr,
    x_range: Optional[Tuple[float, float]] = None,
    show_derivative: bool = True,
    show_second_derivative: bool = False,
    show_critical_points: bool = False,
    show_inflection_points: bool = False,
    show_domain_info: bool = True,
    title: Optional[str] = None,
    num_points: int = DEFAULT_POINTS
) -> plt.Figure:
    """
    Plot a single-variable function with optional derivatives.
    Clean by default - no markers unless explicitly requested.
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    variables = list(get_variables(expr))
    if len(variables) != 1:
        raise ValueError(f"Expected single variable, got {len(variables)}")
    
    var = variables[0]
    
    # Analyze domain
    domain_info = _analyze_domain(expr)
    
    # Determine range
    if x_range is None:
        x_range = _get_safe_range(expr, var)
    
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, num_points)
    
    # Evaluate function
    f = _create_lambda(expr, [var])
    y_raw = _safe_evaluate(f, x)
    
    # Calculate y-limits
    y_min, y_max = _get_smart_ylim(y_raw, None)
    y_range = y_max - y_min
    
    # First detect asymptotes from RAW data (before clipping)
    # An asymptote is where the sign changes and at least one value is large
    asymptote_mask = np.zeros(len(y_raw), dtype=bool)
    for i in range(len(y_raw) - 1):
        if np.isnan(y_raw[i]) or np.isnan(y_raw[i+1]):
            continue
        # Sign change with large values indicates asymptote
        if y_raw[i] * y_raw[i+1] < 0 and (abs(y_raw[i]) > abs(y_max) or abs(y_raw[i+1]) > abs(y_max)):
            asymptote_mask[i] = True
    
    # Now clip values to ylim range
    y = np.clip(y_raw, y_min, y_max)
    
    # Apply NaN at detected asymptotes
    y[asymptote_mask] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=STYLE_CONFIG['figure_size_1d'])
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    # Plot main function
    f_prime_expr = first_derivative(expr, var)
    ax.plot(x, y, color=STYLE_CONFIG['function_color'], 
            linewidth=STYLE_CONFIG['line_width'], 
            label=f'$f({var}) = {latex(expr)}$')
    
    # Plot derivative
    if show_derivative:
        f_prime = _create_lambda(f_prime_expr, [var])
        y_prime_raw = _safe_evaluate(f_prime, x)
        
        # Detect asymptotes from raw data
        asymptote_mask_prime = np.zeros(len(y_prime_raw), dtype=bool)
        for i in range(len(y_prime_raw) - 1):
            if np.isnan(y_prime_raw[i]) or np.isnan(y_prime_raw[i+1]):
                continue
            if y_prime_raw[i] * y_prime_raw[i+1] < 0 and (abs(y_prime_raw[i]) > abs(y_max) or abs(y_prime_raw[i+1]) > abs(y_max)):
                asymptote_mask_prime[i] = True
        
        # Clip and apply NaN at asymptotes
        y_prime = np.clip(y_prime_raw, y_min, y_max)
        y_prime[asymptote_mask_prime] = np.nan
        
        ax.plot(x, y_prime, color=STYLE_CONFIG['derivative_color'],
                linewidth=STYLE_CONFIG['line_width'], linestyle='--',
                label=f"$f'({var}) = {latex(f_prime_expr)}$")
    
    # Set y-limits - matplotlib will clip the display
    ax.set_ylim(y_min, y_max)
    
    # Plot second derivative
    if show_second_derivative:
        f_double_prime_expr = second_derivative(expr, var)
        f_double_prime = _create_lambda(f_double_prime_expr, [var])
        y_double_prime_raw = _safe_evaluate(f_double_prime, x)
        y_double_prime = _break_at_discontinuities(y_double_prime_raw)
        ax.plot(x, y_double_prime, color='#F4A261',
                linewidth=STYLE_CONFIG['line_width'], linestyle=':',
                label=f"$f''({var}) = {latex(f_double_prime_expr)}$")
    
    # Styling
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linestyle='-', linewidth=0.5)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Title
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    else:
        ax.set_title(f'$f({var}) = {latex(expr)}$', fontsize=16, fontweight='bold', pad=15)
    
    ax.set_xlabel(f'${var}$', fontsize=14)
    ax.set_ylabel(f'$f({var})$', fontsize=14)
    
    # Show domain info if there are restrictions
    if show_domain_info and domain_info['has_restrictions']:
        info_text = "Domain: " + "; ".join(domain_info['restrictions'])
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor=STYLE_CONFIG['domain_warning_color'], alpha=0.8))
    
    plt.tight_layout()
    return fig


# =============================================================================
# 2D PLOTTING (Heatmap, Gradient Field)
# =============================================================================

def plot_2d_heatmap(
    expr,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    show_critical_points: bool = False,
    show_domain_info: bool = True,
    title: Optional[str] = None,
    num_points: int = DEFAULT_POINTS_2D,
    cmap: str = 'RdYlBu_r'
) -> plt.Figure:
    """Plot a heatmap. Clean by default - no markers."""
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    variables = sorted(list(get_variables(expr)), key=str)
    if len(variables) != 2:
        raise ValueError(f"Expected 2 variables, got {len(variables)}")
    
    var_x, var_y = variables
    
    # Analyze domain
    domain_info = _analyze_domain(expr)
    
    if x_range is None:
        x_range = _get_safe_range(expr, var_x)
    if y_range is None:
        y_range = _get_safe_range(expr, var_y)
    
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    f = _create_lambda(expr, [var_x, var_y])
    Z = _safe_evaluate(f, X, Y)
    
    # Check for NaN coverage
    nan_percentage = np.sum(np.isnan(Z)) / Z.size * 100
    
    fig, ax = plt.subplots(figsize=STYLE_CONFIG['figure_size_2d'])
    fig.patch.set_facecolor('#FAFAFA')
    
    heatmap = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')
    cbar = plt.colorbar(heatmap, ax=ax, label=f'$f({var_x}, {var_y})$')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlabel(f'${var_x}$', fontsize=14)
    ax.set_ylabel(f'${var_y}$', fontsize=14)
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    else:
        ax.set_title(f'Heatmap: $f({var_x}, {var_y}) = {latex(expr)}$', fontsize=16, fontweight='bold', pad=15)
    
    # Show domain info
    if show_domain_info and (domain_info['has_restrictions'] or nan_percentage > 5):
        info_parts = []
        if domain_info['restrictions']:
            info_parts.extend(domain_info['restrictions'])
        if nan_percentage > 5:
            info_parts.append(f"~{nan_percentage:.0f}% undefined")
        info_text = "Domain: " + "; ".join(info_parts)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor=STYLE_CONFIG['domain_warning_color'], alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_2d_gradient_field(
    expr,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    show_critical_points: bool = False,
    show_domain_info: bool = True,
    title: Optional[str] = None,
    num_arrows: int = 20,
    show_heatmap_background: bool = True
) -> plt.Figure:
    """Plot gradient vector field. Clean by default."""
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    variables = sorted(list(get_variables(expr)), key=str)
    if len(variables) != 2:
        raise ValueError(f"Expected 2 variables, got {len(variables)}")
    
    var_x, var_y = variables
    
    # Analyze domain
    domain_info = _analyze_domain(expr)
    
    if x_range is None:
        x_range = _get_safe_range(expr, var_x)
    if y_range is None:
        y_range = _get_safe_range(expr, var_y)
    
    # Grid for arrows
    x = np.linspace(x_range[0], x_range[1], num_arrows)
    y = np.linspace(y_range[0], y_range[1], num_arrows)
    X, Y = np.meshgrid(x, y)
    
    # Compute gradient
    df_dx = diff(expr, var_x)
    df_dy = diff(expr, var_y)
    
    grad_x_func = _create_lambda(df_dx, [var_x, var_y])
    grad_y_func = _create_lambda(df_dy, [var_x, var_y])
    
    U = _safe_evaluate(grad_x_func, X, Y)
    V = _safe_evaluate(grad_y_func, X, Y)
    
    # Normalize for visibility
    magnitude = np.sqrt(U**2 + V**2)
    magnitude = np.where(magnitude == 0, 1, magnitude)
    U_norm = U / magnitude
    V_norm = V / magnitude
    
    fig, ax = plt.subplots(figsize=STYLE_CONFIG['figure_size_2d'])
    fig.patch.set_facecolor('#FAFAFA')
    
    # Optional heatmap background
    if show_heatmap_background:
        x_fine = np.linspace(x_range[0], x_range[1], 100)
        y_fine = np.linspace(y_range[0], y_range[1], 100)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        f = _create_lambda(expr, [var_x, var_y])
        Z_fine = _safe_evaluate(f, X_fine, Y_fine)
        ax.pcolormesh(X_fine, Y_fine, Z_fine, cmap='gray', alpha=0.3, shading='auto')
    
    # Quiver plot
    quiver = ax.quiver(X, Y, U_norm, V_norm, magnitude, cmap='viridis', alpha=0.8)
    plt.colorbar(quiver, ax=ax, label=r'$|\nabla f|$')
    
    ax.set_xlabel(f'${var_x}$', fontsize=14)
    ax.set_ylabel(f'${var_y}$', fontsize=14)
    ax.grid(True, alpha=STYLE_CONFIG['grid_alpha'])
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    else:
        ax.set_title(rf'Gradient Field: $\nabla f({var_x}, {var_y})$', fontsize=16, fontweight='bold', pad=15)
    
    # Show domain info
    if show_domain_info and domain_info['has_restrictions']:
        info_text = "Domain: " + "; ".join(domain_info['restrictions'])
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor=STYLE_CONFIG['domain_warning_color'], alpha=0.8))
    
    plt.tight_layout()
    return fig


# =============================================================================
# 3D PLOTTING
# =============================================================================

def plot_3d_surface(
    expr,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    show_critical_points: bool = False,
    show_domain_info: bool = True,
    title: Optional[str] = None,
    num_points: int = DEFAULT_POINTS_2D,
    cmap: str = 'viridis',
    alpha: float = 0.85
) -> plt.Figure:
    """Plot a 3D surface. Clean by default - no markers."""
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    variables = sorted(list(get_variables(expr)), key=str)
    if len(variables) != 2:
        raise ValueError(f"Expected 2 variables, got {len(variables)}")
    
    var_x, var_y = variables
    
    # Analyze domain
    domain_info = _analyze_domain(expr)
    
    if x_range is None:
        x_range = _get_safe_range(expr, var_x)
    if y_range is None:
        y_range = _get_safe_range(expr, var_y)
    
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    f = _create_lambda(expr, [var_x, var_y])
    Z = _safe_evaluate(f, X, Y)
    
    # Check for NaN coverage
    nan_percentage = np.sum(np.isnan(Z)) / Z.size * 100
    
    # Create 3D figure
    fig = plt.figure(figsize=STYLE_CONFIG['figure_size_3d'])
    fig.patch.set_facecolor('#FAFAFA')
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha,
                           linewidth=0, antialiased=True)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=f'$f({var_x}, {var_y})$', pad=0.1)
    
    ax.set_xlabel(f'${var_x}$', fontsize=12, labelpad=10)
    ax.set_ylabel(f'${var_y}$', fontsize=12, labelpad=10)
    ax.set_zlabel(f'$f({var_x}, {var_y})$', fontsize=12, labelpad=10)
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    else:
        ax.set_title(f'$f({var_x}, {var_y}) = {latex(expr)}$', fontsize=16, fontweight='bold', pad=20)
    
    # Show domain info as text annotation
    if show_domain_info and (domain_info['has_restrictions'] or nan_percentage > 5):
        info_parts = []
        if domain_info['restrictions']:
            info_parts.extend(domain_info['restrictions'])
        if nan_percentage > 5:
            info_parts.append(f"~{nan_percentage:.0f}% undefined")
        info_text = "Domain: " + "; ".join(info_parts)
        fig.text(0.02, 0.02, info_text, fontsize=9,
                bbox=dict(boxstyle='round', facecolor=STYLE_CONFIG['domain_warning_color'], alpha=0.8))
    
    plt.tight_layout()
    return fig


# =============================================================================
# AUTO-DETECT FUNCTION
# =============================================================================

def plot_function(
    expr,
    plot_type: str = 'auto',
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    **kwargs
) -> plt.Figure:
    """Automatically plot a function based on the number of variables."""
    if isinstance(expr, str):
        expr = parse_expression(expr)
    
    variables = list(get_variables(expr))
    num_vars = len(variables)
    
    if num_vars == 0:
        raise ValueError("Expression has no variables - nothing to plot")
    
    if num_vars == 1:
        return plot_1d(expr, x_range=x_range, **kwargs)
    
    elif num_vars == 2:
        if plot_type == 'auto' or plot_type == '3d':
            return plot_3d_surface(expr, x_range=x_range, y_range=y_range, **kwargs)
        elif plot_type == 'heatmap':
            return plot_2d_heatmap(expr, x_range=x_range, y_range=y_range, **kwargs)
        elif plot_type == 'gradient':
            return plot_2d_gradient_field(expr, x_range=x_range, y_range=y_range, **kwargs)
        else:
            return plot_3d_surface(expr, x_range=x_range, y_range=y_range, **kwargs)
    
    else:
        raise ValueError(f"Cannot plot functions with {num_vars} variables. Maximum is 2.")


def show_plot(fig: plt.Figure = None):
    """Display the plot."""
    plt.show()
