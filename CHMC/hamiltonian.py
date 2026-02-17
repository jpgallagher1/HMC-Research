"""
Description:
    Hamiltonian structures and symplectic operations.
    USE THE CORRECT ENVIRONMENT:  HMC-Research

Author: John Gallagher
Created: 2026-02-16
Last Modified: 2026-02-16
Version: 0.1
"""
from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
from datatypes import QP, HamiltonianState, TargetDensity, MassMatrix, PrecisionMatrix

class Hamiltonian(NamedTuple):
    """
    Hamiltonian(q,p) = U(q) + K(p)
    For standard HMC: 
        U(q) = -log π(q)
        K(p) = 0.5 *  p.T@ M^{-1}@ p
    """
    potential: Callable[[jnp.ndarray], float] # U(q)
    kinetic: Callable[[jnp.ndarray], float] # K(p)

    def energy(self, qp:QP) -> float:
        """total energy H(q,p) = U(q) + K(p)"""
        return self.potential(qp.q) + self.kinetic(qp.p)
    
    def grad_q(self, qp:QP) -> jnp.ndarray:
        """∂H/∂q = ∂U/∂q"""
        return jax.grad(self.potential)(qp.q)
    
    def grad_p(self, qp:QP) ->jnp.ndarray:
        """∂H/∂p = ∂K/∂p"""
        return jax.grad(self.kinetic)(qp.p)
    
    def grad(self, qp:QP) -> QP:
        """Full gradient as QP structure"""
        return QP(q=self.grad_q(qp), p=self.grad_p(qp))

# Symplectic operations
def J_sym(qp: QP) -> QP:
    """
    J is the symplectic Jacobian matrix for Hamiltonians where 
    J = ([[0, I], [-I, 0]])
    """
    return QP(q = qp.p, p = -qp.q)
def qJ_sym(qp:QP) -> QP:
    """
    Update only qside: [p,0]
    Used in position half-step of LF integrator
    """
    return QP(q=qp.p, p=jnp.zeros_like(qp.q))
def pJ_sym(qp: QP) -> QP:
    """
    Update only p side: [0, -q]
    Used in momentum half-step of LF integrator
    """
    return QP(q=jnp.zeros_like(qp.p), p=-qp.q)

# Hamiltonian constructors

def standard_hamiltonian(
    target: TargetDensity,
    mass_inv: MassMatrix
)-> Callable[[QP], float]:
    """
    Hamiltonian(q,p) = U(q) + K(p)
    For standard HMC: 
        U(q) = -log π(q)
        K(p) = 0.5 *  p.T@ M^{-1}@ p
    """

    def hamiltonian(qp:QP) -> float:
        U = -jnp.log(target(qp.q))
        K = 0.5*jnp.dot(qp.p, mass_inv@qp.p)
        return U+K
    return hamiltonian

def gaussian_hamiltonian(
    precision: PrecisionMatrix,
    mass_inv: MassMatrix
)-> Callable[[QP], float]:
    """
    Hamiltonian(q,p) = U(q) + K(p)
    For standard HMC: 
        U(q) = -log π(q)
        K(p) = 0.5 *  p.T@ M^{-1}@ p
    """

    def hamiltonian(qp:QP) -> float:
        U = 0.5 * jnp.dot(qp.q, precision @ qp.q)
        K = 0.5 * jnp.dot(qp.p, mass_inv @ qp.p)
        return U+K
    return hamiltonian

def hamiltonian_from_flat(H_flat: Callable[[jnp.ndarray], float]) -> Callable[[QP], float]:
    """
    Convert a flat-array Hamiltonian to QP-based Hamiltonian
    
    Useful for legacy code compatibility.
    """
    def hamiltonian(qp: QP) -> float:
        return H_flat(qp.to_array())
    
    return hamiltonian