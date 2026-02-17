"""
Description:
    Core data structures for CHMC.
    USE THE CORRECT ENVIRONMENT:  HMC-Research

Author: John Gallagher
Created: 2026-02-16
Last Modified: 2026-02-16
Version: 0.1

All modules import from here to ensure type consistency and avoid indexing bugs.
"""
from typing import NamedTuple, Callable
import jax.numpy as jnp

class QP(NamedTuple):
    """Phase space state(q,p)"""
    q: jnp.ndarray # position
    p: jnp.ndarray # momentum

    @property
    def dim(self) -> int:
        """Dimension of configuration space"""
        return self.q.shape[0]
    def to_array(self) -> jnp.ndarray:
        """Convert to flat array [q,p] for compatability with old code"""
        return jnp.concatenate([self.q, self.p])
    @classmethod
    def from_array(cls, arr: jnp.ndarray):
        """Convert from flat array[q,p]"""
        dim = arr.shape[0]//2
        return cls(q=arr[:dim], p=arr[dim:])

class HamiltonianState(NamedTuple):
    """State for Hamiltonian evaluation"""
    qp: QP
    energy: float
    grad: QP # May create an issue downstream

class IntegratorState(NamedTuple):
    """State during FPI Newton iteration"""
    qp: QP
    residual: QP # F(qp) tolerance
    step_size: float # adaptive step size during integration
    n_iter: int
    converged: bool
    residual_norm: float

class SamplerState(NamedTuple):
    """MCMC sampler state"""
    qp: QP 
    deltaH: float 
    accepted: bool

class SamplerOutput(NamedTuple):
    samples: jnp.ndarray # (n_samples, dim) - positions only
    deltaH: jnp.ndarray # Energy differences
    accept_rate: float # bool? 

class IntegratorConfig (NamedTuple):
    """Configuration for numerical integrators"""
    τ: float # time-step size
    N: int # Number of integration steps
    tol: float = 1e-2 # Tolerance of implicit method
    max_iter: int = 3 # Max Newton iter

# Type aliases for clarity
TargetDensity = Callable[[jnp.ndarray], float]
MassMatrix = jnp.ndarray
PrecisionMatrix = jnp.ndarray