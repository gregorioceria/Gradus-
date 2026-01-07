"""
Gradus Week 3 - Plotting Module

This module provides visualization functions for:
- 1D plots: f(x) with derivatives and critical points
- 2D plots: Heatmaps, gradient fields for f(x,y)
- 3D plots: Surface plots for f(x,y)
"""

from .plotting import (
    plot_1d,
    plot_2d_heatmap,
    plot_2d_gradient_field,
    plot_3d_surface,
    plot_function,  # Auto-detect 1D or 2D/3D
)

