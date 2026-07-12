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
import jax
import jax.numpy as jnp
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QP:
## partially generated with claude input ##
    x: jnp.ndarray                          # the ONLY leaf — shape (..., 2*dim)

    # views, for reading/debugging only
    @property
    def dim(self): return self.x.shape[-1] // 2

    @property
    def q(self):   return self.x[..., : self.dim]
    @property
    def p(self):   return self.x[..., self.dim :]

    @classmethod
    def from_qp(cls, q, p):                  # ergonomic constructor
        return cls(jnp.concatenate([q, p], axis=-1))

    # the object *is* the flat vector
    def __add__(s, o):  return QP(s.x + (o.x if isinstance(o, QP) else o))
    def __sub__(s, o):  return QP(s.x - (o.x if isinstance(o, QP) else o))
    def __mul__(s, o):  return QP(s.x * (o.x if isinstance(o, QP) else o))
    def __rmul__(s, o): return QP(o * s.x)
    def __neg__(s):     return QP(-s.x)
    def symplectic(s):  return QP.from_qp(s.p, -s.q)   # J·x = [p, -q]

    def to_array(self) -> jnp.ndarray:
        """Convert to flat array [q,p] for compatability with old code"""
        return self.x
    @classmethod
    def from_array(cls, arr: jnp.ndarray):
        """Convert from flat array[q,p]"""
        # dim = arr.shape[0]//2 # depricated and largely unused now. 
        return cls(arr)

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
    T: float # final time T = N*τ
    N: int # Number of integration steps = int(ceil(T/τ))
    tol: float = 1e-2 # Tolerance of implicit method
    max_iter: int = 3 # Max Newton iter
    constant_p: bool = False # used in sampler
    integrator: str = 'midptNewtonFPI'
    AA_m: int = 5
    AA_beta: float=1
    n_pts: int = 4 # THIS IS NEW ####
    trajectory: bool = False # THIS IS NEW ####
    gen_gauss: bool = False
    debug: bool = False

# Type aliases for clarity
TargetDensity = Callable[[jnp.ndarray], float]
MassMatrix = jnp.ndarray
PrecisionMatrix = jnp.ndarray