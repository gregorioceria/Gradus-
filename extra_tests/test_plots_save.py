"""
Test script - Clean plots + domain warnings.
"""
import os
import sys
sys.path.insert(0, '.')

os.makedirs('output', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gradus.week3 import (
    plot_1d,
    plot_2d_heatmap,
    plot_2d_gradient_field,
    plot_3d_surface,
)

print("=" * 60)
print("  GENERATING CLEAN PLOTS + DOMAIN TESTS")
print("=" * 60)

# =============================================================================
# 1D PLOTS - Standard
# =============================================================================
print("\n[1] 1D Standard Plots...")

fig = plot_1d("x^3 - 3*x", show_derivative=True)
fig.savefig('output/1d_polynomial.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 1d_polynomial.png")

fig = plot_1d("sin(x)", show_derivative=True)
fig.savefig('output/1d_sin.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 1d_sin.png")

fig = plot_1d("exp(-x^2)", show_derivative=True)
fig.savefig('output/1d_gaussian.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 1d_gaussian.png")

# =============================================================================
# 1D PLOTS - Domain Limited (show warnings)
# =============================================================================
print("\n[2] 1D Domain Limited (with warnings)...")

fig = plot_1d("log(x)", show_derivative=True, show_domain_info=True)
fig.savefig('output/1d_log_domain.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 1d_log_domain.png (shows domain warning)")

fig = plot_1d("sqrt(x)", show_derivative=True, show_domain_info=True)
fig.savefig('output/1d_sqrt_domain.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 1d_sqrt_domain.png (shows domain warning)")

fig = plot_1d("1/x", show_derivative=True, show_domain_info=True, x_range=(-5, 5))
fig.savefig('output/1d_inverse_domain.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 1d_inverse_domain.png (shows singularity warning)")

fig = plot_1d("tan(x)", show_derivative=False, show_domain_info=True, x_range=(-3, 3))
fig.savefig('output/1d_tan_domain.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 1d_tan_domain.png (shows asymptote warning)")

# =============================================================================
# 2D HEATMAPS - Standard
# =============================================================================
print("\n[3] 2D Heatmaps...")

fig = plot_2d_heatmap("x^2 + y^2")
fig.savefig('output/2d_heatmap_paraboloid.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 2d_heatmap_paraboloid.png")

fig = plot_2d_heatmap("x^2 - y^2")
fig.savefig('output/2d_heatmap_saddle.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 2d_heatmap_saddle.png")

# =============================================================================
# 2D HEATMAPS - Domain Limited
# =============================================================================
print("\n[4] 2D Heatmaps with Domain Issues...")

fig = plot_2d_heatmap("log(x + y)", show_domain_info=True)
fig.savefig('output/2d_heatmap_log_domain.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 2d_heatmap_log_domain.png (shows domain warning)")

fig = plot_2d_heatmap("1/(x^2 + y^2)", show_domain_info=True, x_range=(-5, 5), y_range=(-5, 5))
fig.savefig('output/2d_heatmap_singularity.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 2d_heatmap_singularity.png (shows % undefined)")

# =============================================================================
# 2D GRADIENT FIELDS
# =============================================================================
print("\n[5] 2D Gradient Fields...")

fig = plot_2d_gradient_field("x^2 + y^2")
fig.savefig('output/2d_gradient_paraboloid.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 2d_gradient_paraboloid.png")

fig = plot_2d_gradient_field("x^2 - y^2")
fig.savefig('output/2d_gradient_saddle.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 2d_gradient_saddle.png")

# =============================================================================
# 3D SURFACES - Standard
# =============================================================================
print("\n[6] 3D Surfaces...")

fig = plot_3d_surface("x^2 + y^2")
fig.savefig('output/3d_paraboloid.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 3d_paraboloid.png")

fig = plot_3d_surface("x^2 - y^2")
fig.savefig('output/3d_saddle.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 3d_saddle.png")

fig = plot_3d_surface("sin(x)*cos(y)", x_range=(-5, 5), y_range=(-5, 5))
fig.savefig('output/3d_sincos.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 3d_sincos.png")

fig = plot_3d_surface("exp(-(x^2 + y^2))")
fig.savefig('output/3d_gaussian.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 3d_gaussian.png")

# =============================================================================
# 3D SURFACES - Domain Limited
# =============================================================================
print("\n[7] 3D Surfaces with Domain Issues...")

fig = plot_3d_surface("log(x*y)", show_domain_info=True, x_range=(0.1, 5), y_range=(0.1, 5))
fig.savefig('output/3d_log_domain.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 3d_log_domain.png (shows domain warning)")

fig = plot_3d_surface("sqrt(x + y)", show_domain_info=True)
fig.savefig('output/3d_sqrt_domain.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print("  -> 3d_sqrt_domain.png (shows domain warning)")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 60)
print("  DONE!")
print("=" * 60)
print("\nGenerated files:")
for f in sorted(os.listdir('output')):
    if f.endswith('.png'):
        domain_tag = " [DOMAIN]" if "domain" in f or "singularity" in f else ""
        print(f"  - output/{f}{domain_tag}")
