"""
Description:
    MCMC diagnostics and metrics.
    USE THE CORRECT ENVIRONMENT:  HMC-Research

Author: John Gallagher
Created: 2026-02-16
Last Modified: 2026-02-16
Version: 0.1
"""
import jax
from pathlib import Path
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
import numpy as np
from typing import Tuple

def compute_accept_rate(samples: Tuple) -> float:
    """
    Compute acceptance rate from samples.
    
    Args:
        samples: Output from sampler
        
    Returns:
        Acceptance rate in [0, 1]
    """
    _, _, accepted = samples
    return jnp.mean(accepted)


def cov(X):
    Xμ = jnp.mean(X, axis = 0)
    n=X.shape[0]
    return (X - Xμ).T@(X-Xμ)/(n-1)

def cov_diag(X, sample_axis = 0, ddof = 1):
    return jnp.var(X, axis=sample_axis, ddof = 1)

def masked_cov_diag(X, mask, sample_axis=0, ddof=1):
    """
    X    : (N, d)
    mask : (N,) bool

    Returns
    -------
    (d,) diagonal of the covariance matrix computed over accepted samples.
    """
    w = mask.astype(X.dtype)
    n = jnp.sum(w)
    mean = jnp.sum(X * w[:, None], axis=sample_axis) / n
    var = jnp.sum(w[:, None] * (X - mean) ** 2, axis=sample_axis)
    return var / (n - ddof)

def maxtracediff(X,Y) -> float:
    x = jnp.diag(X)
    y = jnp.diag(Y)
    return jnp.max(jnp.abs(x-y))

def maxdiff(X,Y) -> float:
    return jnp.max(jnp.abs(X-Y))

def random_1d_projection(key, target_dim: int) -> tuple:
    x = jr.uniform(key, target_dim)
    x_normed = x/jnp.linalg.norm(x)
    return jnp.linalg.outer(x_normed,x_normed)

def gen_2grid(min, max, N = 201, dim=2):
    """
    gen meshgrid of shape (N*N, 2)
    """
    x = np.linspace(min, max, N)
    y = np.linspace(min, max, N)
    Gx, Gy = np.meshgrid(x, y)
    grid = np.stack([Gx, Gy], axis=-1).reshape(-1, dim)  # (NxN, 2)
    return grid

def gen_random_gridpts_mat_pdf(precision, NxN2_grid, m_points = 100):
    """
    Subset of random gridpoints to generate pdf for w_1_nd metric, using precision matrix
    """
    AXY_grid = NxN2_grid @ np.array(precision).T              # (NxN, 2)
    Z_grid   = np.einsum('...i,...i->...', NxN2_grid, AXY_grid)  
    pdf_grid = np.exp(-0.5 * Z_grid)                    
    pdf_grid /= pdf_grid.sum()                          

    idx = np.random.choice(len(NxN2_grid), size=m_points, p=pdf_grid)
    pdf_samples = NxN2_grid[idx]   
    return pdf_samples

def gen_1d_pdf(key, target, target_dim=2, min=-2, max=2, m_points=101):
    grid = gen_2grid(min, max, m_points, target_dim)
    Z = target(grid)
    rand_vec = jr.uniform(key, target_dim)
    rand_vec_normed = jnp.expand_dims(rand_vec/jnp.linalg.norm(rand_vec), 1)
    projmat_rand_vec = rand_vec_normed@rand_vec_normed.T@grid
    

def gen_w1_random_1d_pdf(key, target, dim= 2, min= -2, max= 2, m_points= 101):
    """
    generate 1d vector for w1 metric. only works for gaussian right now
    """
    scale = jnp.expand_dims(jnp.linspace(min, max, m_points), 1)
    rand_vec = jr.uniform(key, dim)
    rand_vec_normed = rand_vec/jnp.linalg.norm(rand_vec)
    x = scale*rand_vec_normed
    AX = x@np.array(target).T
    Z = np.einsum('...i,...i->...', x, AX)
    pdf_line = np.exp(-0.5 * Z)
    pdf_line /= pdf_line.sum()  
    return pdf_line

def gen_random_gridpts_nongauss_pdf(target, NxN2_grid, m_points = 101):
    """
    Subset of random gridpoints to generate pdf for w_1_nd metric, generic target
    """
    assert NxN2_grid.shape[-1] ==2, 'Shape must be (NxN, 2)'
    AXY_grid = NxN2_grid 
    pdf_grid   = jax.vmap(target)(AXY_grid)  
    pdf_grid /= pdf_grid.sum()                          

    idx = np.random.choice(len(NxN2_grid), size=m_points, p=pdf_grid)
    pdf_samples = NxN2_grid[idx]   
    return pdf_samples

def gen_cdf1D_marginal(key, target, dim = 2, m_points=101):
    """
    Generate a 1d cdf along a random vector through the origin, respecting the marginal distribution.
    
    For use in wasserstein 1 metric:
    
    """
    proj_n = random_1d_projection(key, dim)
    
    return
def max_τ(target_mat):
    return 2/jnp.sqrt(jnp.max(jnp.linalg.eigvals(target_mat)))

def load_result(base, method, tau, T, length, run):
    """
    navigating the file path generated from the forloops. 
    result = load_result(
        base,
        method = "AA",
        tau = 2**-1,
        T = 1.0,
        length = 1000,
        run = 3,
    )

    q = result["q"]
    deltaHs = result["deltaHs"]
    accepted = result["accepted"]
    runtime = result["runtime"]

    """
    path = (
        Path(base)
        / method
        / f"tau_{float(tau):.12g}"
        / f"T_{float(T):.12g}"
        / f"len_{int(length)}"
        / f"run_{run}.npz"
    )
    return jnp.load(path)