#!/usr/bin/env python3
"""
Run Demo: Generate All Visualizations
======================================

This script generates all figures for the OU process tutorial.
Run from the project root: python run_demo.py
"""

import os
import numpy as np

# Create figures directory
os.makedirs('figures', exist_ok=True)

print("="*60)
print("  ORNSTEIN-UHLENBECK PROCESS TUTORIAL")
print("  Generating Visualizations...")
print("="*60)

from visualizations import (
    plot_mean_reversion_demo,
    plot_theta_comparison,
    plot_sigma_comparison,
    plot_stationary_distribution,
    plot_2d_trajectories,
    plot_anisotropic_comparison,
    plot_parameter_space
)

# =============================================================================
# FIGURE 1: Core Mean Reversion Behavior
# =============================================================================
print("\n[1/7] Mean reversion demo...")
print("      Showing how paths are pulled toward equilibrium")
fig = plot_mean_reversion_demo(
    theta=0.3,
    mu=0.0,
    sigma=0.4,
    x0=4.0,
    T=25.0,
    n_paths=50,
    save_path='figures/01_mean_reversion.png'
)

# =============================================================================
# FIGURE 2: Effect of Mean-Reversion Speed (theta)
# =============================================================================
print("\n[2/7] Theta comparison...")
print("      Comparing slow vs fast reversion")
fig = plot_theta_comparison(
    thetas=[0.1, 0.5, 2.0],
    mu=0.0,
    sigma=0.5,
    x0=5.0,
    T=20.0,
    save_path='figures/02_theta_comparison.png'
)

# =============================================================================
# FIGURE 3: Effect of Volatility (sigma)
# =============================================================================
print("\n[3/7] Sigma comparison...")
print("      Comparing low vs high volatility")
fig = plot_sigma_comparison(
    sigmas=[0.1, 0.5, 1.5],
    theta=0.3,
    mu=0.0,
    x0=3.0,
    T=25.0,
    save_path='figures/03_sigma_comparison.png'
)

# =============================================================================
# FIGURE 4: Stationary Distribution
# =============================================================================
print("\n[4/7] Stationary distribution...")
print("      Showing convergence to N(μ, σ²/2θ)")
fig = plot_stationary_distribution(
    theta=0.4,
    mu=2.0,
    sigma=0.7,
    x0=-3.0,
    T=40.0,
    n_paths=500,
    save_path='figures/04_stationary_dist.png'
)

# =============================================================================
# FIGURE 5: 2D Trajectories
# =============================================================================
print("\n[5/7] 2D trajectories...")
print("      Visualizing mean reversion in the plane")
fig = plot_2d_trajectories(
    theta_x=0.3,
    theta_y=0.3,
    mu=np.array([0.0, 0.0]),
    sigma=0.5,
    x0=np.array([6.0, 5.0]),
    T=35.0,
    n_paths=25,
    save_path='figures/05_2d_trajectories.png'
)

# =============================================================================
# FIGURE 6: Anisotropic Comparison
# =============================================================================
print("\n[6/7] Anisotropic comparison...")
print("      Different reversion speeds per dimension")
fig = plot_anisotropic_comparison(
    eigenvalue_pairs=[
        (0.3, 0.3),   # Isotropic
        (0.8, 0.2),   # Faster in X
        (0.2, 0.8),   # Faster in Y
        (1.5, 0.1),   # Very anisotropic
    ],
    mu=np.array([0.0, 0.0]),
    sigma=0.5,
    x0=np.array([5.0, 5.0]),
    T=30.0,
    n_paths=15,
    save_path='figures/06_anisotropic.png'
)

# =============================================================================
# FIGURE 7: Parameter Space
# =============================================================================
print("\n[7/7] Parameter space exploration...")
print("      Relaxation time and stationary variance heatmaps")
fig = plot_parameter_space(
    save_path='figures/07_parameter_space.png'
)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("  ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
for i, name in enumerate([
    "01_mean_reversion.png",
    "02_theta_comparison.png", 
    "03_sigma_comparison.png",
    "04_stationary_dist.png",
    "05_2d_trajectories.png",
    "06_anisotropic.png",
    "07_parameter_space.png"
], 1):
    print(f"  [{i}] figures/{name}")

print("\nNext steps:")
print("  • Open figures/ to view the plots")
print("  • Import visualizations.py for custom plots")
print("  • See README.md for documentation")
print("="*60)
