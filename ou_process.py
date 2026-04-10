"""
Ornstein-Uhlenbeck Process: Intuitive Simulation & Visualization
=================================================================

A clean, educational implementation of the OU process for understanding
mean-reversion dynamics in stochastic systems.

dX_t = θ(μ - X_t)dt + σ dW_t

where:
    θ (theta) = mean-reversion speed (how fast we return to equilibrium)
    μ (mu)    = long-term mean (equilibrium level)  
    σ (sigma) = volatility (noise intensity)
    W_t       = Wiener process (Brownian motion)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from dataclasses import dataclass


# =============================================================================
# CORE SIMULATION
# =============================================================================

def simulate_ou_1d(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    T: float = 10.0,
    dt: float = 0.01,
    n_paths: int = 100,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate 1D Ornstein-Uhlenbeck process via Euler-Maruyama.
    
    Parameters
    ----------
    theta : float
        Mean-reversion speed (larger = faster return to mu)
    mu : float
        Long-term mean (equilibrium level)
    sigma : float
        Volatility (noise intensity)
    x0 : float
        Initial position
    T : float
        Total simulation time
    dt : float
        Time step size
    n_paths : int
        Number of independent paths to simulate
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    t : (n_steps,) array
        Time points
    X : (n_paths, n_steps) array
        Simulated trajectories
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_steps = int(T / dt) + 1
    t = np.linspace(0, T, n_steps)
    X = np.zeros((n_paths, n_steps))
    X[:, 0] = x0
    
    sqrt_dt = np.sqrt(dt)
    
    for i in range(1, n_steps):
        dW = np.random.randn(n_paths) * sqrt_dt
        drift = theta * (mu - X[:, i-1]) * dt
        diffusion = sigma * dW
        X[:, i] = X[:, i-1] + drift + diffusion
    
    return t, X


def simulate_ou_2d(
    A: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    x0: np.ndarray,
    T: float = 10.0,
    dt: float = 0.01,
    n_paths: int = 100,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate 2D Ornstein-Uhlenbeck process.
    
    dX_t = A(μ - X_t)dt + σ dW_t
    
    Parameters
    ----------
    A : (2, 2) array
        Mean-reversion matrix (eigenvalues control relaxation in each direction)
    mu : (2,) array
        Long-term mean vector
    sigma : float
        Isotropic noise intensity
    x0 : (2,) array
        Initial position
        
    Returns
    -------
    t : (n_steps,) array
    X : (n_paths, n_steps, 2) array
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_steps = int(T / dt) + 1
    t = np.linspace(0, T, n_steps)
    X = np.zeros((n_paths, n_steps, 2))
    X[:, 0, :] = x0
    
    sqrt_dt = np.sqrt(dt)
    
    for i in range(1, n_steps):
        dW = np.random.randn(n_paths, 2) * sqrt_dt
        diff = mu - X[:, i-1, :]  # (n_paths, 2)
        drift = diff @ A.T * dt   # (n_paths, 2)
        diffusion = sigma * dW
        X[:, i, :] = X[:, i-1, :] + drift + diffusion
    
    return t, X


# =============================================================================
# THEORETICAL PROPERTIES
# =============================================================================

@dataclass
class OUProperties:
    """Theoretical properties of the OU process."""
    theta: float
    mu: float
    sigma: float
    relaxation_time: float      # τ = 1/θ
    stationary_variance: float  # σ²/(2θ)
    stationary_std: float       # sqrt(stationary_variance)


def compute_ou_properties(theta: float, mu: float, sigma: float) -> OUProperties:
    """
    Compute theoretical properties of a 1D OU process.
    
    Key formulas:
    - Relaxation time: τ = 1/θ (time to decay by factor of e)
    - Stationary variance: Var(X_∞) = σ²/(2θ)
    - Stationary distribution: X_∞ ~ N(μ, σ²/(2θ))
    """
    tau = 1 / theta
    stat_var = sigma**2 / (2 * theta)
    stat_std = np.sqrt(stat_var)
    
    return OUProperties(
        theta=theta,
        mu=mu,
        sigma=sigma,
        relaxation_time=tau,
        stationary_variance=stat_var,
        stationary_std=stat_std
    )


def theoretical_mean(t: np.ndarray, x0: float, mu: float, theta: float) -> np.ndarray:
    """
    Expected value E[X_t] for OU process.
    
    E[X_t] = μ + (x0 - μ)e^(-θt)
    """
    return mu + (x0 - mu) * np.exp(-theta * t)


def theoretical_variance(t: np.ndarray, theta: float, sigma: float) -> np.ndarray:
    """
    Variance Var(X_t) for OU process starting at deterministic x0.
    
    Var(X_t) = (σ²/2θ)(1 - e^(-2θt))
    """
    return (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t))


# =============================================================================
# PLOTTING UTILITIES (using publication-quality style)
# =============================================================================

def setup_style():
    """Configure matplotlib for beautiful, publication-quality plots."""
    plt.rcParams.update({
        # Figure
        'figure.facecolor': '#0a0a0f',
        'figure.edgecolor': '#0a0a0f',
        'figure.figsize': (12, 7),
        'figure.dpi': 150,
        
        # Axes
        'axes.facecolor': '#0a0a0f',
        'axes.edgecolor': '#3a3a4a',
        'axes.labelcolor': '#e0e0e0',
        'axes.titlecolor': '#ffffff',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Grid
        'grid.color': '#2a2a3a',
        'grid.alpha': 0.5,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        
        # Ticks
        'xtick.color': '#a0a0a0',
        'ytick.color': '#a0a0a0',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        
        # Text
        'text.color': '#e0e0e0',
        'font.family': 'sans-serif',
        'font.size': 11,
        
        # Legend
        'legend.facecolor': '#1a1a2a',
        'legend.edgecolor': '#3a3a4a',
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,
        
        # Lines
        'lines.linewidth': 1.5,
        
        # Savefig
        'savefig.facecolor': '#0a0a0f',
        'savefig.edgecolor': '#0a0a0f',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })


# Color palette - vibrant colors on dark background
COLORS = {
    'primary': '#00d4aa',      # Teal/cyan
    'secondary': '#ff6b6b',    # Coral red  
    'tertiary': '#a855f7',     # Purple
    'quaternary': '#fbbf24',   # Amber
    'mean_line': '#ff6b6b',    # For mu line
    'theory': '#ffffff',       # White for theoretical curves
    'confidence': '#00d4aa',   # For confidence bands
}


def get_trajectory_colors(n: int, cmap_name: str = 'plasma') -> np.ndarray:
    """Get array of colors for multiple trajectories."""
    cmap = plt.cm.get_cmap(cmap_name)
    return cmap(np.linspace(0.2, 0.9, n))


def get_time_colors(n_steps: int, cmap_name: str = 'viridis') -> np.ndarray:
    """Get colors that progress with time."""
    cmap = plt.cm.get_cmap(cmap_name)
    return cmap(np.linspace(0, 1, n_steps))
