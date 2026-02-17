"""
Description:
    Target distribution generators.
    USE THE CORRECT ENVIRONMENT:  HMC-Research

Author: John Gallagher
Created: 2026-02-16
Last Modified: 2026-02-16
Version: 0.1
"""
import jax.numpy as jnp
from datatypes import TargetDensity, PrecisionMatrix

def gen_gaussian(
        dim: int = 2,
        precision_matrix: PrecisionMatrix = None,
        cov: jnp.ndarray = None
) -> TargetDensity:
    if precision_matrix is not None and cov is not None:
        raise ValueError(
            "Please supply either a precision_matrix or a cov, not both"
        )
    
    if precision_matrix is None and cov is not None:
        precision_matrix = jnp.linalg.inv(cov)
    
    if precision_matrix is None and cov is None:
        precision_matrix = jnp.eye(dim)
    def target(q: jnp.ndarray) -> float:
        """Gaussian target density (unnormalized)"""
        return jnp.exp(-0.5 * jnp.dot(q, precision_matrix @ q))
    
    return target

def gen_perturb_precision(
        dim: int = 2,
        perturbation: float = 0.05
) -> PrecisionMatrix:
    prec = jnp.diag(jnp.ones(dim))
    prec += perturbation * jnp.diag(jnp.ones(dim-1), k=-1 )
    prec += perturbation * jnp.diag(jnp.ones(dim-1), k=1 )
    return prec