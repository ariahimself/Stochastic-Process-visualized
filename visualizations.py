"""
Ornstein-Uhlenbeck Process: Visualization Module
=================================================

Beautiful, educational visualizations for understanding OU dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection
from typing import Optional, List, Tuple
from ou_process import (
    simulate_ou_1d, simulate_ou_2d,
    compute_ou_properties, theoretical_mean, theoretical_variance,
    setup_style, COLORS, get_trajectory_colors, get_time_colors
)


# =============================================================================
# 1D VISUALIZATIONS
# =============================================================================

def plot_mean_reversion_demo(
    theta: float = 0.5,
    mu: float = 0.0,
    sigma: float = 0.3,
    x0: float = 3.0,
    T: float = 20.0,
    n_paths: int = 50,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Demonstrate mean-reversion: paths pulled toward equilibrium.
    
    This is the key intuition - the OU process is like a spring
    pulling particles back to the mean.
    """
    setup_style()
    
    t, X = simulate_ou_1d(theta, mu, sigma, x0, T=T, n_paths=n_paths)
    props = compute_ou_properties(theta, mu, sigma)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot sample paths with alpha gradient
    colors = get_trajectory_colors(n_paths, 'cool')
    for i in range(n_paths):
        ax.plot(t, X[i], color=colors[i], alpha=0.4, linewidth=0.8)
    
    # Highlight a few paths
    for i in [0, n_paths//3, 2*n_paths//3]:
        ax.plot(t, X[i], color=COLORS['primary'], alpha=0.9, linewidth=2)
    
    # Theoretical mean (exponential decay to mu)
    mean_theory = theoretical_mean(t, x0, mu, theta)
    ax.plot(t, mean_theory, color=COLORS['theory'], linewidth=3, 
            linestyle='--', label=f'E[X(t)] = μ + (x₀-μ)e^(-θt)')
    
    # Equilibrium level
    ax.axhline(mu, color=COLORS['mean_line'], linewidth=2, linestyle='-',
               label=f'μ = {mu} (equilibrium)')
    
    # Confidence bands (±2 std from stationary distribution)
    std_theory = np.sqrt(theoretical_variance(t, theta, sigma))
    upper = mean_theory + 2*std_theory
    lower = mean_theory - 2*std_theory
    ax.fill_between(t, lower, upper, alpha=0.15, color=COLORS['confidence'],
                    label='±2σ confidence band')
    
    # Annotations
    ax.annotate(f'Start: x₀ = {x0}', xy=(0, x0), xytext=(1, x0 + 0.5),
                fontsize=11, color=COLORS['primary'],
                arrowprops=dict(arrowstyle='->', color=COLORS['primary']))
    
    # Relaxation time marker
    tau = props.relaxation_time
    y_tau = theoretical_mean(np.array([tau]), x0, mu, theta)[0]
    ax.axvline(tau, color=COLORS['quaternary'], linewidth=1.5, linestyle=':',
               alpha=0.7)
    ax.annotate(f'τ = 1/θ = {tau:.1f}s\n(relaxation time)', 
                xy=(tau, y_tau), xytext=(tau + 2, y_tau + 0.8),
                fontsize=10, color=COLORS['quaternary'],
                arrowprops=dict(arrowstyle='->', color=COLORS['quaternary']))
    
    # Labels and title
    ax.set_xlabel('Time (t)', fontsize=13)
    ax.set_ylabel('X(t)', fontsize=13)
    ax.set_title(f'Mean Reversion in OU Process\nθ = {theta}, μ = {mu}, σ = {sigma}',
                 fontsize=15, pad=20)
    ax.legend(loc='upper right', fontsize=11)
    
    # Add parameter box
    textstr = f'Parameters:\n θ = {theta} (reversion speed)\n μ = {mu} (long-term mean)\n σ = {sigma} (volatility)\n\nDerived:\n τ = {tau:.2f}s (relaxation time)\n σ_∞ = {props.stationary_std:.3f} (stationary std)'
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#1a1a2a', 
                     edgecolor=COLORS['primary'], alpha=0.9)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=box_props, family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    
    return fig


def plot_theta_comparison(
    thetas: List[float] = [0.1, 0.5, 2.0],
    mu: float = 0.0,
    sigma: float = 0.5,
    x0: float = 5.0,
    T: float = 20.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare different mean-reversion speeds (θ).
    
    Key insight: Higher θ = faster convergence to μ.
    """
    setup_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['quaternary']]
    
    for idx, (theta, color) in enumerate(zip(thetas, colors)):
        ax = axes[idx]
        t, X = simulate_ou_1d(theta, mu, sigma, x0, T=T, n_paths=30, seed=42)
        props = compute_ou_properties(theta, mu, sigma)
        
        # Plot paths
        for i in range(30):
            ax.plot(t, X[i], color=color, alpha=0.25, linewidth=0.8)
        
        # Mean path
        ax.plot(t, X.mean(axis=0), color=color, linewidth=2.5, label='Sample mean')
        
        # Theoretical mean
        mean_theory = theoretical_mean(t, x0, mu, theta)
        ax.plot(t, mean_theory, color=COLORS['theory'], linewidth=2,
                linestyle='--', label='E[X(t)]')
        
        # Equilibrium
        ax.axhline(mu, color=COLORS['mean_line'], linewidth=1.5, linestyle='-',
                   alpha=0.7)
        
        # Relaxation time
        tau = props.relaxation_time
        ax.axvline(tau, color='white', linewidth=1, linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('X(t)' if idx == 0 else '', fontsize=11)
        ax.set_title(f'θ = {theta}\nτ = {tau:.1f}s', fontsize=13, color=color)
        ax.set_ylim(-3, 7)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    fig.suptitle('Effect of Mean-Reversion Speed (θ)\nHigher θ → Faster Return to Equilibrium',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    
    return fig


def plot_sigma_comparison(
    sigmas: List[float] = [0.1, 0.5, 1.5],
    theta: float = 0.3,
    mu: float = 0.0,
    x0: float = 3.0,
    T: float = 25.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare different volatilities (σ).
    
    Key insight: Higher σ = wider fluctuations around μ.
    """
    setup_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['quaternary']]
    
    for idx, (sigma, color) in enumerate(zip(sigmas, colors)):
        ax = axes[idx]
        t, X = simulate_ou_1d(theta, mu, sigma, x0, T=T, n_paths=30, seed=42)
        props = compute_ou_properties(theta, mu, sigma)
        
        # Plot paths
        for i in range(30):
            ax.plot(t, X[i], color=color, alpha=0.3, linewidth=0.8)
        
        # Equilibrium
        ax.axhline(mu, color=COLORS['mean_line'], linewidth=2, linestyle='-')
        
        # Stationary std bands
        stat_std = props.stationary_std
        ax.axhline(mu + 2*stat_std, color='white', linewidth=1, linestyle='--', alpha=0.5)
        ax.axhline(mu - 2*stat_std, color='white', linewidth=1, linestyle='--', alpha=0.5)
        ax.fill_between(t, mu - 2*stat_std, mu + 2*stat_std, 
                        alpha=0.1, color=color)
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('X(t)' if idx == 0 else '', fontsize=11)
        ax.set_title(f'σ = {sigma}\nStationary std = {stat_std:.2f}', 
                     fontsize=13, color=color)
        ax.set_ylim(-6, 8)
    
    fig.suptitle(f'Effect of Volatility (σ) with θ = {theta}\nHigher σ → Wider Fluctuations',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    
    return fig


def plot_stationary_distribution(
    theta: float = 0.5,
    mu: float = 2.0,
    sigma: float = 0.8,
    x0: float = -3.0,
    T: float = 50.0,
    n_paths: int = 500,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Show convergence to stationary distribution.
    
    Key insight: After long time, X(t) ~ N(μ, σ²/2θ) regardless of x₀.
    """
    setup_style()
    
    t, X = simulate_ou_1d(theta, mu, sigma, x0, T=T, n_paths=n_paths)
    props = compute_ou_properties(theta, mu, sigma)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), 
                             gridspec_kw={'width_ratios': [2, 1]})
    
    # Left: trajectories
    ax1 = axes[0]
    colors = get_trajectory_colors(min(50, n_paths), 'plasma')
    for i in range(min(50, n_paths)):
        ax1.plot(t, X[i], color=colors[i], alpha=0.3, linewidth=0.6)
    
    # Mean and confidence bands
    mean_theory = theoretical_mean(t, x0, mu, theta)
    std_theory = np.sqrt(theoretical_variance(t, theta, sigma))
    ax1.plot(t, mean_theory, color=COLORS['theory'], linewidth=2.5, 
             linestyle='--', label='E[X(t)]')
    ax1.fill_between(t, mean_theory - 2*std_theory, mean_theory + 2*std_theory,
                     alpha=0.2, color=COLORS['confidence'])
    ax1.axhline(mu, color=COLORS['mean_line'], linewidth=2, label=f'μ = {mu}')
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('X(t)', fontsize=12)
    ax1.set_title('Trajectories Converging to Equilibrium', fontsize=13)
    ax1.legend(loc='upper right')
    
    # Right: histogram of final values
    ax2 = axes[1]
    final_values = X[:, -1]
    
    ax2.hist(final_values, bins=40, density=True, orientation='horizontal',
             alpha=0.7, color=COLORS['primary'], edgecolor='white', linewidth=0.5)
    
    # Theoretical stationary distribution
    y_grid = np.linspace(final_values.min() - 0.5, final_values.max() + 0.5, 200)
    stat_pdf = (1 / (props.stationary_std * np.sqrt(2*np.pi))) * \
               np.exp(-0.5 * ((y_grid - mu) / props.stationary_std)**2)
    ax2.plot(stat_pdf, y_grid, color=COLORS['theory'], linewidth=2.5,
             label='N(μ, σ²/2θ)')
    
    ax2.axhline(mu, color=COLORS['mean_line'], linewidth=2, linestyle='-')
    ax2.axhline(mu + 2*props.stationary_std, color='white', linewidth=1, 
                linestyle='--', alpha=0.5)
    ax2.axhline(mu - 2*props.stationary_std, color='white', linewidth=1,
                linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Density', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_title('Stationary\nDistribution', fontsize=13)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(ax1.get_ylim())
    
    fig.suptitle(f'Convergence to Stationary Distribution\nX_∞ ~ N({mu}, {props.stationary_variance:.3f})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 2D VISUALIZATIONS
# =============================================================================

def plot_2d_trajectories(
    theta_x: float = 0.3,
    theta_y: float = 0.3,
    mu: np.ndarray = None,
    sigma: float = 0.5,
    x0: np.ndarray = None,
    T: float = 30.0,
    n_paths: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize 2D OU process trajectories.
    
    Shows the "spring-like" attraction to equilibrium in 2D.
    """
    setup_style()
    
    if mu is None:
        mu = np.array([0.0, 0.0])
    if x0 is None:
        x0 = np.array([5.0, 4.0])
    
    A = np.diag([theta_x, theta_y])
    t, X = simulate_ou_2d(A, mu, sigma, x0, T=T, n_paths=n_paths, dt=0.02)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectories with time-varying color
    n_steps = X.shape[1]
    time_colors = get_time_colors(n_steps, 'plasma')
    
    for p in range(n_paths):
        # Create line segments with time coloring
        points = X[p, :, :].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, colors=time_colors[:-1], 
                           linewidth=1.2, alpha=0.6)
        ax.add_collection(lc)
    
    # Mark start and equilibrium
    ax.scatter(x0[0], x0[1], c=COLORS['primary'], s=300, marker='o',
               edgecolors='white', linewidths=2, zorder=10, label=f'Start ({x0[0]}, {x0[1]})')
    ax.scatter(mu[0], mu[1], c=COLORS['mean_line'], s=400, marker='*',
               edgecolors='white', linewidths=2, zorder=10, label=f'μ ({mu[0]}, {mu[1]})')
    
    # Confidence ellipse (stationary distribution)
    stat_std_x = sigma / np.sqrt(2 * theta_x)
    stat_std_y = sigma / np.sqrt(2 * theta_y)
    
    theta_grid = np.linspace(0, 2*np.pi, 100)
    for n_std in [1, 2]:
        ellipse_x = mu[0] + n_std * stat_std_x * np.cos(theta_grid)
        ellipse_y = mu[1] + n_std * stat_std_y * np.sin(theta_grid)
        ax.plot(ellipse_x, ellipse_y, color='white', linewidth=1.5,
                linestyle='--', alpha=0.5, 
                label=f'±{n_std}σ contour' if n_std == 2 else '')
    
    ax.set_xlabel('X₁', fontsize=13)
    ax.set_ylabel('X₂', fontsize=13)
    ax.set_title(f'2D OU Process Trajectories\nθ_x = {theta_x}, θ_y = {theta_y}, σ = {sigma}',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_aspect('equal')
    
    # Set axis limits with some padding
    all_x = X[:, :, 0].flatten()
    all_y = X[:, :, 1].flatten()
    padding = 1.0
    ax.set_xlim(min(all_x.min(), mu[0] - 3*stat_std_x) - padding,
                max(all_x.max(), mu[0] + 3*stat_std_x) + padding)
    ax.set_ylim(min(all_y.min(), mu[1] - 3*stat_std_y) - padding,
                max(all_y.max(), mu[1] + 3*stat_std_y) + padding)
    
    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='plasma', 
                                norm=plt.Normalize(vmin=0, vmax=T))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Time')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    
    return fig


def plot_anisotropic_comparison(
    eigenvalue_pairs: List[Tuple[float, float]] = None,
    mu: np.ndarray = None,
    sigma: float = 0.5,
    x0: np.ndarray = None,
    T: float = 30.0,
    n_paths: int = 15,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare isotropic vs anisotropic mean-reversion.
    
    Key insight: Different eigenvalues create directional preferences.
    When θ_x >> θ_y, the process is "stiff" in X direction but "loose" in Y.
    """
    setup_style()
    
    if eigenvalue_pairs is None:
        eigenvalue_pairs = [
            (0.3, 0.3),   # Isotropic
            (0.8, 0.2),   # Faster in X
            (0.2, 0.8),   # Faster in Y
            (1.5, 0.1),   # Very anisotropic
        ]
    
    if mu is None:
        mu = np.array([0.0, 0.0])
    if x0 is None:
        x0 = np.array([5.0, 5.0])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    titles = ['Isotropic', 'Faster reversion in X', 
              'Faster reversion in Y', 'Strongly anisotropic']
    
    for idx, ((theta_x, theta_y), ax, title) in enumerate(
            zip(eigenvalue_pairs, axes.flatten(), titles)):
        
        A = np.diag([theta_x, theta_y])
        t, X = simulate_ou_2d(A, mu, sigma, x0, T=T, n_paths=n_paths, dt=0.02, seed=42)
        
        # Time-colored trajectories
        n_steps = X.shape[1]
        time_colors = get_time_colors(n_steps, 'plasma')
        
        for p in range(n_paths):
            points = X[p, :, :].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=time_colors[:-1],
                               linewidth=1.0, alpha=0.5)
            ax.add_collection(lc)
        
        # Markers
        ax.scatter(x0[0], x0[1], c=COLORS['primary'], s=200, marker='o',
                   edgecolors='white', linewidths=2, zorder=10)
        ax.scatter(mu[0], mu[1], c=COLORS['mean_line'], s=250, marker='*',
                   edgecolors='white', linewidths=2, zorder=10)
        
        # Confidence ellipse
        stat_std_x = sigma / np.sqrt(2 * theta_x)
        stat_std_y = sigma / np.sqrt(2 * theta_y)
        theta_grid = np.linspace(0, 2*np.pi, 100)
        
        for n_std in [2]:
            ellipse_x = mu[0] + n_std * stat_std_x * np.cos(theta_grid)
            ellipse_y = mu[1] + n_std * stat_std_y * np.sin(theta_grid)
            ax.plot(ellipse_x, ellipse_y, color='white', linewidth=2,
                    linestyle='--', alpha=0.6)
        
        ax.set_xlabel('X₁', fontsize=11)
        ax.set_ylabel('X₂', fontsize=11)
        ax.set_title(f'{title}\nθ_x = {theta_x}, θ_y = {theta_y}', fontsize=12)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 8)
        ax.set_ylim(-5, 8)
    
    fig.suptitle('Anisotropic Mean-Reversion: Effect of Different θ per Dimension',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# REGIME DIAGRAM
# =============================================================================

def plot_parameter_space(
    theta_range: np.ndarray = None,
    sigma_range: np.ndarray = None,
    mu: float = 0.0,
    x0: float = 5.0,
    T: float = 30.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize behavior across (θ, σ) parameter space.
    
    Shows how relaxation time and stationary variance depend on parameters.
    """
    setup_style()
    
    if theta_range is None:
        theta_range = np.linspace(0.1, 2.0, 30)
    if sigma_range is None:
        sigma_range = np.linspace(0.1, 2.0, 30)
    
    # Compute theoretical quantities
    THETA, SIGMA = np.meshgrid(theta_range, sigma_range)
    TAU = 1 / THETA  # Relaxation time
    STAT_VAR = SIGMA**2 / (2 * THETA)  # Stationary variance
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Relaxation time
    ax1 = axes[0]
    im1 = ax1.pcolormesh(THETA, SIGMA, TAU, shading='auto', cmap='viridis')
    contours1 = ax1.contour(THETA, SIGMA, TAU, levels=[0.5, 1, 2, 5, 10],
                            colors='white', linewidths=1.5)
    ax1.clabel(contours1, inline=True, fontsize=10, fmt='τ=%.1f')
    plt.colorbar(im1, ax=ax1, label='Relaxation Time τ = 1/θ')
    ax1.set_xlabel('θ (mean-reversion speed)', fontsize=12)
    ax1.set_ylabel('σ (volatility)', fontsize=12)
    ax1.set_title('Relaxation Time\nτ = 1/θ (independent of σ)', fontsize=13)
    
    # Right: Stationary variance  
    ax2 = axes[1]
    im2 = ax2.pcolormesh(THETA, SIGMA, STAT_VAR, shading='auto', cmap='magma')
    contours2 = ax2.contour(THETA, SIGMA, STAT_VAR, levels=[0.1, 0.5, 1, 2, 5],
                            colors='white', linewidths=1.5)
    ax2.clabel(contours2, inline=True, fontsize=10, fmt='Var=%.1f')
    plt.colorbar(im2, ax=ax2, label='Stationary Variance σ²/2θ')
    ax2.set_xlabel('θ (mean-reversion speed)', fontsize=12)
    ax2.set_ylabel('σ (volatility)', fontsize=12)
    ax2.set_title('Stationary Variance\nVar(X_∞) = σ²/2θ', fontsize=13)
    
    fig.suptitle('OU Process Parameter Space', fontsize=15, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("OU PROCESS VISUALIZATION SUITE")
    print("="*60)
    
    # Generate all figures
    print("\n[1/6] Mean reversion demo...")
    plot_mean_reversion_demo(save_path='figures/01_mean_reversion.png')
    
    print("\n[2/6] Theta comparison...")
    plot_theta_comparison(save_path='figures/02_theta_comparison.png')
    
    print("\n[3/6] Sigma comparison...")
    plot_sigma_comparison(save_path='figures/03_sigma_comparison.png')
    
    print("\n[4/6] Stationary distribution...")
    plot_stationary_distribution(save_path='figures/04_stationary_dist.png')
    
    print("\n[5/6] 2D trajectories...")
    plot_2d_trajectories(save_path='figures/05_2d_trajectories.png')
    
    print("\n[6/6] Anisotropic comparison...")
    plot_anisotropic_comparison(save_path='figures/06_anisotropic.png')
    
    print("\n" + "="*60)
    print("All figures saved to figures/")
    print("="*60)
