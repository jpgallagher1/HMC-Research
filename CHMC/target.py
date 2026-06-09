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

def gen_perturb_mat(
        dim: int = 2,
        perturbation: float = 0.05
) -> PrecisionMatrix:
    prec = jnp.diag(jnp.ones(dim))
    prec += perturbation * jnp.diag(jnp.ones(dim-1), k=-1 )
    prec += perturbation * jnp.diag(jnp.ones(dim-1), k=1 )
    return prec
def gen_2D_perturb_vals(x):
    """input desired κ(A, np.inf), output is perturb val"""
    return (x-1)/(x+1)

def max_τ(target_mat):
    return 2/jnp.sqrt(jnp.max(jnp.linalg.eigvals(target_mat)))

def banana(q, a=1, b=100): 
    """Rosebrock function aka banana"""
    assert q.shape == (2,), 'q must be 2 dim'
    q1, q2 = q[0], q[1]
    log_density = -((a - q1)**2 + b * (q2 - q1**2)**2)
    return jnp.exp(log_density)

def himmelblau(q):
    assert q.shape == (2,), 'q must be 2 dim'
    x,y = q[0], q[1]
    log_density = -((x**2+y-11)**2+(x+y**2-7)**2)
    return jnp.exp(log_density)

def fourpeaks(q):
    "coefficient is computed numerically using desmos... https://www.desmos.com/calculator/uk3ixhdkah"

    assert q.shape == (2,), 'q must be 2 dim'
    x,y = q[0], q[1]
    log_density = -0.5*((x**2+y-2)**2+(x+y**2-2)**2)
    return 1/3.90992630367*jnp.exp(log_density)