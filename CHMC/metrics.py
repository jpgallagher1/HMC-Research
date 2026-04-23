"""
Description:
    MCMC diagnostics and metrics.
    USE THE CORRECT ENVIRONMENT:  HMC-Research

Author: John Gallagher
Created: 2026-02-16
Last Modified: 2026-02-16
Version: 0.1
"""
import jax.numpy as jnp
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

def maxtracediff(X,Y) -> float:
    x = jnp.diag(X)
    y = jnp.diag(Y)
    return jnp.max(jnp.abs(x-y))