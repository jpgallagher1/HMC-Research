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

def cov(X):
    Xμ = jnp.mean(X, axis = 0)
    n=X.shape[0]
    return (X - Xμ).T@(X-Xμ)/(n-1)

def maxdiagdiff(X,Y):
    x = np.diag(X)
    y = np.diag(Y)
    return np.max(np.abs(x-y))