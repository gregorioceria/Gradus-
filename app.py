"""
Gradus - Symbolic Derivative Calculator
Web Application powered by Streamlit
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sympy import latex, Symbol, symbols, N, pi, E, oo, zoo, nan, I, diff
from typing import Dict, Any, Optional, List

# Import Gradus modules
import sys
sys.path.insert(0, '.')

from gradus.week1.parsing import parse_expression, get_variables, ParsingError
from gradus.week1.single_var import first_derivative, second_derivative, third_derivative, all_derivatives
from gradus.week1.domain import get_domain_restrictions, check_evaluation_point
from gradus.week2.multi_var import (
    partial_derivative, all_partial_derivatives, gradient, hessian,
    gradient_at_point, hessian_at_point, classify_critical_point
)
from gradus.week3.plotting import plot_1d, plot_2d_heatmap, plot_2d_gradient_field, plot_3d_surface

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Gradus - Derivative Calculator",
    page_icon="∇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - MATHEMATICAL STYLE
# =============================================================================

st.markdown("""
<style>
    /* Import mathematical fonts and Material Icons */
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff !important;
        font-family: 'Source Serif 4', 'Crimson Pro', 'Times New Roman', serif !important;
    }
    
    /* All text - mathematical serif font */
    p, span, label, div, li {
        color: #f0f0f0 !important;
        font-family: 'Source Serif 4', 'Crimson Pro', 'Georgia', serif !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
        border-right: 1px solid #2d2d4a;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #e0e0e0 !important;
    }
    
    /* Headers - elegant gradient */
    h1, h2, h3, h4 {
        background: linear-gradient(90deg, #a855f7, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Crimson Pro', 'Georgia', serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }
    
    /* Strong/Bold text */
    strong, b {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Input fields - elegant */
    .stTextInput > div > div > input {
        background-color: #1a1a2e !important;
        border: 1px solid #4a4a6a !important;
        color: #ffffff !important;
        border-radius: 8px;
        font-size: 1.1rem;
        font-family: 'Source Serif 4', serif !important;
        padding: 0.6rem 1rem;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #7777aa !important;
        font-style: italic;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.25) !important;
    }
    
    /* Labels */
    .stTextInput > label, .stSelectbox > label, .stCheckbox > label {
        color: #d0d0e0 !important;
        font-weight: 500;
        font-size: 0.95rem;
        font-family: 'Source Serif 4', serif !important;
    }
    
    /* Buttons - elegant gradient */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #06b6d4) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        font-family: 'Source Serif 4', serif !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.35) !important;
    }
    
    /* Select boxes - closed state */
    .stSelectbox > div > div {
        background-color: #1a1a2e !important;
        border: 1px solid #4a4a6a !important;
        border-radius: 8px;
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div > div {
        color: #ffffff !important;
        font-family: 'Source Serif 4', serif !important;
    }
    
    /* DROPDOWN MENU - FORCE BLACK TEXT */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="popover"] * {
        color: #000000 !important;
    }
    
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="menu"] * {
        color: #000000 !important;
        font-family: 'Source Serif 4', serif !important;
    }
    
    [data-baseweb="menu"] li {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #e8e8f0 !important;
        color: #000000 !important;
    }
    
    [role="listbox"] {
        background-color: #ffffff !important;
    }
    
    [role="listbox"] * {
        color: #000000 !important;
    }
    
    [role="option"] {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    [role="option"]:hover {
        background-color: #e8e8f0 !important;
    }
    
    [role="option"] span {
        color: #000000 !important;
    }
    
    /* Checkbox */
    .stCheckbox > label > span {
        color: #f0f0f0 !important;
        font-family: 'Source Serif 4', serif !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a2e !important;
        border: 1px solid #3d3d5c !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-family: 'Source Serif 4', serif !important;
    }
    
    /* HTML details styling */
    details summary {
        list-style: none;
    }
    
    details summary::-webkit-details-marker {
        display: none;
    }
    
    details[open] summary {
        margin-bottom: 8px;
    }
    
    /* LaTeX - elegant white and bigger */
    .katex {
        font-size: 1.6em !important;
        color: #ffffff !important;
    }
    
    .katex * {
        color: #ffffff !important;
    }
    
    /* Metric cards - hide them */
    [data-testid="stMetricValue"] {
        display: none !important;
    }
    
    [data-testid="stMetricLabel"] {
        display: none !important;
    }
    
    /* Divider */
    hr {
        border-color: #3a3a5a !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Info/Warning/Error boxes */
    .stInfo, .stWarning, .stError, .stSuccess {
        color: #ffffff !important;
        font-family: 'Source Serif 4', serif !important;
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Number input */
    .stNumberInput > div > div > input {
        background-color: #1a1a2e !important;
        border: 1px solid #4a4a6a !important;
        color: #ffffff !important;
        font-family: 'Source Serif 4', serif !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_point_value(value_str: str) -> float:
    """Parse evaluation point value (supports fractions, pi, e)."""
    value_str = value_str.strip().lower()
    
    if value_str == 'pi':
        return float(pi.evalf())
    elif value_str == 'e':
        return float(E.evalf())
    
    if '/' in value_str:
        parts = value_str.split('/')
        if len(parts) == 2:
            num = parse_point_value(parts[0])
            den = parse_point_value(parts[1])
            return num / den
    
    if 'pi' in value_str:
        value_str = value_str.replace('pi', str(float(pi.evalf())))
    if 'e' in value_str and not value_str.replace('.', '').replace('-', '').isdigit():
        if 'e+' not in value_str and 'e-' not in value_str:
            value_str = value_str.replace('e', str(float(E.evalf())))
    
    return float(eval(value_str))


def parse_multivar_point(point_str: str, variables: List[Symbol]) -> Dict[Symbol, float]:
    """Parse multivariable point from string."""
    point = {}
    point_str = point_str.strip()
    
    if '=' in point_str:
        for part in point_str.replace(' ', '').split(','):
            if '=' in part:
                var_name, val = part.split('=')
                point[Symbol(var_name.strip())] = parse_point_value(val.strip())
    else:
        values = [v.strip() for v in point_str.split(',')]
        for var, val in zip(variables, values):
            point[var] = parse_point_value(val)
    
    return point


def format_number(val: float) -> str:
    """Format number for LaTeX display."""
    if abs(val) < 1e-10:
        return "0"
    if abs(val - round(val)) < 1e-10:
        return str(int(round(val)))
    return f"{val:.4g}"


def matrix_to_latex_large(matrix: np.ndarray) -> str:
    """Convert numpy matrix to LaTeX pmatrix format with large size."""
    rows = []
    for i in range(matrix.shape[0]):
        row = " & ".join([format_number(matrix[i, j]) for j in range(matrix.shape[1])])
        rows.append(row)
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    try:
        st.image("gradus_logo.jpeg", use_container_width=True)
    except:
        st.markdown("# ∇ Gradus")
    
    st.markdown("---")
    
    mode = st.selectbox(
        "Calculation Mode",
        ["Single Variable", "Multivariable"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("""
<details style="cursor: pointer;">
<summary style="color: #e0e0e0; font-size: 1.1em; font-weight: 600; padding: 8px 0; font-family: 'Source Serif 4', serif;">▶ Help</summary>
<div style="padding: 10px 0; color: #d0d0d0;">
<p><strong>Trig:</strong> sin, cos, tan, cot</p>
<p><strong>Inverse:</strong> asin, acos, atan (or arcsin, arccos, arctan)</p>
<p><strong>Other:</strong> log, ln, exp, sqrt, abs</p>
<p><strong>Operators:</strong> +, -, *, /, ^</p>
<p><strong>Constants:</strong> pi, e</p>
<p><strong>Examples:</strong></p>
<ul style="margin-left: 20px;">
<li>x^3 - 3*x + 2</li>
<li>atan(x) or arctan(x)</li>
<li>x^2 + y^2</li>
</ul>
</div>
</details>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #8888aa; font-size: 0.85em;'>"
        "Made with ❤️ for Software Engineering<br>"
        "Bocconi University 2025"
        "</div>",
        unsafe_allow_html=True
    )


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown("<h1 style='text-align: center; font-size: 3.2rem;'>∇ Gradus</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaaacc; margin-bottom: 2rem; font-size: 1.3rem; font-style: italic;'>Symbolic Derivative Calculator</p>", unsafe_allow_html=True)

# =============================================================================
# SINGLE VARIABLE MODE
# =============================================================================

if mode == "Single Variable":
    st.markdown("### Single Variable Derivatives")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        function_input = st.text_input(
            "Enter function f(x):",
            placeholder="e.g., x^3 - 3*x + 2",
            key="single_var_input"
        )
    
    with col2:
        derivative_order = st.selectbox(
            "Order:",
            ["1st", "2nd", "3rd", "All"],
            index=3
        )
    
    col3, col4, col5 = st.columns([2, 1, 1])
    
    with col3:
        eval_point = st.text_input(
            "Evaluate at x = (optional):",
            placeholder="e.g., 2, pi/4, 1/2",
            key="eval_point"
        )
    
    with col4:
        show_plot = st.checkbox("Show Graph", value=False)
    
    with col5:
        if show_plot:
            show_deriv_plot = st.checkbox("Show f'(x)", value=True)
        else:
            show_deriv_plot = False
    
    if st.button("Calculate", key="calc_single", use_container_width=True):
        if function_input:
            try:
                expr = parse_expression(function_input)
                variables = list(get_variables(expr))
                
                if len(variables) == 0:
                    st.warning("Expression is a constant. All derivatives are 0.")
                elif len(variables) > 1:
                    st.error(f"Found {len(variables)} variables. Use Multivariable mode.")
                else:
                    var = variables[0]
                    
                    st.markdown("---")
                    
                    if derivative_order == "All":
                        results = all_derivatives(expr, var)
                        
                        st.markdown("### Results")
                        
                        cols = st.columns(4)
                        
                        with cols[0]:
                            st.markdown("**f(x)**")
                            st.latex(latex(results['f']))
                        
                        with cols[1]:
                            st.markdown("**f′(x)**")
                            st.latex(latex(results['f_prime']))
                        
                        with cols[2]:
                            st.markdown("**f″(x)**")
                            st.latex(latex(results['f_double_prime']))
                        
                        with cols[3]:
                            st.markdown("**f‴(x)**")
                            st.latex(latex(results['f_triple_prime']))
                    
                    else:
                        order_map = {"1st": 1, "2nd": 2, "3rd": 3}
                        order = order_map[derivative_order]
                        
                        if order == 1:
                            deriv = first_derivative(expr, var)
                            deriv_name = "f′(x)"
                        elif order == 2:
                            deriv = second_derivative(expr, var)
                            deriv_name = "f″(x)"
                        else:
                            deriv = third_derivative(expr, var)
                            deriv_name = "f‴(x)"
                        
                        st.markdown("### Result")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**f(x)**")
                            st.latex(latex(expr))
                        with col_b:
                            st.markdown(f"**{deriv_name}**")
                            st.latex(latex(deriv))
                    
                    restrictions = get_domain_restrictions(expr)
                    if restrictions:
                        st.info(f"**Domain:** {'; '.join(restrictions)}")
                    
                    if eval_point:
                        try:
                            x_val = parse_point_value(eval_point)
                            domain_ok, domain_msg = check_evaluation_point(expr, var, x_val)
                            
                            if not domain_ok:
                                st.error(f"{domain_msg}")
                            else:
                                st.markdown("---")
                                st.markdown(f"### Evaluation at x = {x_val:.6g}")
                                
                                if derivative_order == "All":
                                    f_val = float(expr.subs(var, x_val).evalf())
                                    fp_val = float(results['f_prime'].subs(var, x_val).evalf()) if results['f_prime'] != 0 else 0.0
                                    fpp_val = float(results['f_double_prime'].subs(var, x_val).evalf()) if results['f_double_prime'] != 0 else 0.0
                                    fppp_val = float(results['f_triple_prime'].subs(var, x_val).evalf()) if results['f_triple_prime'] != 0 else 0.0
                                    
                                    st.latex(rf"f({x_val:.4g}) = {format_number(f_val)}")
                                    st.latex(rf"f'({x_val:.4g}) = {format_number(fp_val)}")
                                    st.latex(rf"f''({x_val:.4g}) = {format_number(fpp_val)}")
                                    st.latex(rf"f'''({x_val:.4g}) = {format_number(fppp_val)}")
                                else:
                                    f_val = float(expr.subs(var, x_val).evalf())
                                    d_val = float(deriv.subs(var, x_val).evalf())
                                    st.latex(rf"f({x_val:.4g}) = {format_number(f_val)}")
                                    st.latex(rf"{deriv_name.replace('(x)', '')}({x_val:.4g}) = {format_number(d_val)}")
                        
                        except Exception as e:
                            st.error(f"Error evaluating: {e}")
                    
                    if show_plot:
                        st.markdown("---")
                        st.markdown("### Graph")
                        with st.spinner("Generating plot..."):
                            fig = plot_1d(expr, show_derivative=show_deriv_plot, show_domain_info=True)
                            st.pyplot(fig)
                            plt.close(fig)
            
            except ParsingError as e:
                st.error(f"Parsing Error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a function.")


# =============================================================================
# MULTIVARIABLE MODE
# =============================================================================

elif mode == "Multivariable":
    st.markdown("### Multivariable Analysis")
    
    function_input = st.text_input(
        "Enter function f(x, y, ...):",
        placeholder="e.g., x^2 + y^2, x*y*z",
        key="multi_var_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Complete Analysis", "Partial Derivatives", "Gradient", "Hessian Matrix"]
        )
    
    with col2:
        show_plot_multi = st.checkbox("Show Graph", value=False)
    
    with col3:
        if show_plot_multi:
            plot_type_multi = st.selectbox("Plot Type:", ["3D Surface", "Heatmap", "Gradient Field"])
        else:
            plot_type_multi = "3D Surface"
    
    eval_point_multi = st.text_input(
        "Evaluate at point (optional):",
        placeholder="e.g., x=1, y=2  or  1, 2",
        key="eval_point_multi"
    )
    
    if st.button("Analyze", key="calc_multi", use_container_width=True):
        if function_input:
            try:
                expr = parse_expression(function_input)
                variables = sorted(list(get_variables(expr)), key=str)
                
                if len(variables) < 2:
                    st.error("Need at least 2 variables. Use Single Variable mode.")
                elif len(variables) > 6:
                    st.error("Maximum 6 variables supported.")
                else:
                    st.markdown("---")
                    
                    st.markdown("### Results")
                    st.markdown("**Function:**")
                    st.latex(f"f({', '.join(str(v) for v in variables)}) = {latex(expr)}")
                    
                    if analysis_type in ["Partial Derivatives", "Complete Analysis"]:
                        st.markdown("---")
                        st.markdown("#### Partial Derivatives")
                        
                        partials = all_partial_derivatives(expr)
                        cols = st.columns(min(len(variables), 3))
                        
                        for i, (var, deriv) in enumerate(partials.items()):
                            with cols[i % 3]:
                                st.latex(rf"\frac{{\partial f}}{{\partial {var}}} = {latex(deriv)}")
                    
                    if analysis_type in ["Gradient", "Complete Analysis"]:
                        st.markdown("---")
                        st.markdown("#### Gradient")
                        
                        grad_components = [diff(expr, v) for v in variables]
                        grad_latex = r"\nabla f = \begin{pmatrix} " + r" \\ ".join([latex(g) for g in grad_components]) + r" \end{pmatrix}"
                        st.latex(grad_latex)
                    
                    if analysis_type in ["Hessian Matrix", "Complete Analysis"]:
                        st.markdown("---")
                        st.markdown("#### Hessian Matrix")
                        
                        hess_matrix, hess_vars = hessian(expr)
                        rows = []
                        for i in range(hess_matrix.shape[0]):
                            row = " & ".join([latex(hess_matrix[i, j]) for j in range(hess_matrix.shape[1])])
                            rows.append(row)
                        hess_latex = r"H = \begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"
                        st.latex(hess_latex)
                    
                    # Numerical Evaluation
                    if eval_point_multi:
                        try:
                            st.markdown("---")
                            st.markdown("#### Numerical Evaluation")
                            
                            point = parse_multivar_point(eval_point_multi, variables)
                            
                            missing = [str(v) for v in variables if v not in point]
                            if missing:
                                st.error(f"Missing values for: {', '.join(missing)}")
                            else:
                                # Build point string for display
                                point_str = ", ".join([f"{v}={point[v]:.4g}" for v in variables])
                                
                                # Function value
                                f_val = float(expr.subs(point).evalf())
                                st.latex(rf"f({point_str}) = {format_number(f_val)}")
                                
                                # Gradient at point
                                grad_vals = [float(diff(expr, v).subs(point).evalf()) for v in variables]
                                grad_vec = r"\begin{pmatrix} " + r" \\ ".join([format_number(v) for v in grad_vals]) + r" \end{pmatrix}"
                                st.latex(rf"\nabla f({point_str}) = {grad_vec}")
                                
                                # Hessian at point
                                hess_matrix, hess_vars = hessian(expr)
                                hess_vals = np.array([[float(hess_matrix[i, j].subs(point).evalf()) 
                                                      for j in range(hess_matrix.shape[1])] 
                                                     for i in range(hess_matrix.shape[0])])
                                
                                hess_mat_latex = matrix_to_latex_large(hess_vals)
                                st.latex(rf"H({point_str}) = {hess_mat_latex}")
                                
                                # Critical point classification
                                grad_norm = np.linalg.norm(grad_vals)
                                if grad_norm < 1e-6:
                                    eigenvalues = np.linalg.eigvalsh(hess_vals)
                                    if np.all(eigenvalues > 0):
                                        cp_type = "Local Minimum"
                                    elif np.all(eigenvalues < 0):
                                        cp_type = "Local Maximum"
                                    elif np.any(eigenvalues == 0):
                                        cp_type = "Degenerate (inconclusive)"
                                    else:
                                        cp_type = "Saddle Point"
                                    st.success(f"**Critical Point Classification:** {cp_type}")
                        
                        except Exception as e:
                            st.error(f"Error evaluating at point: {e}")
                    
                    # Plot
                    if show_plot_multi and len(variables) == 2:
                        st.markdown("---")
                        st.markdown("### Graph")
                        with st.spinner("Generating plot..."):
                            if plot_type_multi == "3D Surface":
                                fig = plot_3d_surface(expr, show_domain_info=True)
                            elif plot_type_multi == "Heatmap":
                                fig = plot_2d_heatmap(expr, show_domain_info=True)
                            else:
                                fig = plot_2d_gradient_field(expr, show_domain_info=True)
                            st.pyplot(fig)
                            plt.close(fig)
                    elif show_plot_multi and len(variables) > 2:
                        st.warning("Cannot plot functions with more than 2 variables.")
            
            except ParsingError as e:
                st.error(f"Parsing Error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a function.")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #8888aa; font-size: 0.95em; font-style: italic;'>"
    "Gradus v1.0 — Software Engineering Project — Bocconi University 2025"
    "</div>",
    unsafe_allow_html=True
)
