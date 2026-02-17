"""
Test suite for CHMC implementation.

Compares numerical integrators (leapfrog and implicit midpoint) 
against their analytical solutions for a simple harmonic oscillator.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Callable
from datatypes import QP, IntegratorConfig
from target import gen_gaussian
from hamiltonian import gaussian_hamiltonian
from integrator import gen_leapfrog, gen_midptFPI
# from jaxtyping import Array, Float

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# ============================================================================
# Analytical Solutions
# ============================================================================

def leapfrog_analytic(x: np.ndarray, tau: float) -> np.ndarray:
    """
    Analytical leapfrog step for simple harmonic oscillator
    """
    LF_step = np.array([
        [1 - tau**2/2, tau - tau**3/4],
        [-tau, 1 - tau**2/2]
    ])
    return LF_step @ x


def implicit_midpoint_analytic(x: np.ndarray, tau: float) -> np.ndarray:
    """
    Analytical implicit midpoint step for simple harmonic oscillator
    """
    factor = 1 / (1 + tau**2/4)
    IM_step = factor * np.array([
        [1 - tau**2/4, tau],
        [-tau, 1 - tau**2/4]
    ])
    return IM_step @ x


# ============================================================================
# Tests
# ============================================================================

def test_leapfrog():
    """Test leapfrog integrator against analytical solution"""
    print("=" * 70)
    print("Testing Leapfrog Integrator")
    print("=" * 70)
    
    # Setup
    dim = 1
    tau = 0.1
    N = 1
    mass_inv = jnp.eye(dim)
    target = gen_gaussian(dim=1)
    
    # Create Hamiltonian and gradient
    hamiltonian = gaussian_hamiltonian(mass_inv, target)
    gradH = jax.jit(jax.grad(hamiltonian))
    
    # Wrap grad to work with QP
    def gradH_qp(qp: QP) -> QP:
        grad_flat = gradH(qp)
        return grad_flat
    
    # Create integrator
    config = IntegratorConfig(τ=tau, N=N)
    one_lf = gen_leapfrog(gradH_qp, config)
    
    # Initial condition
    key = jax.random.PRNGKey(1)
    x0_flat = jax.random.normal(key, shape=(2 * dim,))
    x0 = QP.from_array(x0_flat)
    
    print(f"Initial state: q={x0.q}, p={x0.p}")
    print(f"Initial state (flat): {x0_flat}")
    
    # Numerical integration
    x_lf = one_lf(x0)
    x_lf_flat = x_lf.to_array()
    
    # Analytical solution
    x_analytic = leapfrog_analytic(np.array(x0_flat), tau)
    
    # Compare
    print(f"\nLeapfrog (numerical): {x_lf_flat}")
    print(f"Leapfrog (analytic) : {x_analytic}")
    print(f"Difference          : {np.abs(x_lf_flat - x_analytic)}")
    
    # Energy conservation
    H0 = hamiltonian(x0)
    H_lf = hamiltonian(x_lf)
    H_analytic = hamiltonian(QP.from_array(jnp.array(x_analytic)))
    
    print(f"\nEnergy H(x0)        : {H0:.8f}")
    print(f"Energy H(x_lf)      : {H_lf:.8f}")
    print(f"Energy H(x_analytic): {H_analytic:.8f}")
    print(f"Energy error (LF)   : {np.abs(H_lf - H0):.2e}")
    
    # Assert accuracy
    assert np.allclose(x_lf_flat, x_analytic, atol=1e-10), "Leapfrog test failed!"
    print("\n✓ Leapfrog test PASSED")


def test_implicit_midpoint():
    """Test implicit midpoint integrator against analytical solution"""
    print("\n" + "=" * 70)
    print("Testing Implicit Midpoint Integrator")
    print("=" * 70)
    
    # Setup
    dim = 1
    tau = 0.1
    N = 1
    mass_inv = jnp.eye(dim)
    target = gen_gaussian
    
    # Create Hamiltonian and gradient
    hamiltonian = gaussian_hamiltonian(mass_inv, target)
    gradH = jax.jit(jax.grad(hamiltonian))
    
    # Wrap grad to work with QP
    def gradH_qp(qp: QP) -> QP:
        grad_flat = gradH(qp)
        return grad_flat
    
    # Create integrator
    config = IntegratorConfig(tau=tau, N=N, tol=1e-6, max_iter=10)
    one_midpoint = gen_midptFPI(gradH_qp, config)
    
    # Initial condition
    key = jax.random.PRNGKey(1)
    x0_flat = jax.random.normal(key, shape=(2 * dim,))
    x0 = QP.from_array(x0_flat)
    
    print(f"Initial state: q={x0.q}, p={x0.p}")
    print(f"Initial state (flat): {x0_flat}")
    
    # Numerical integration
    x_mid = one_midpoint(x0)
    x_mid_flat = x_mid.to_array()
    
    # Analytical solution
    x_analytic = implicit_midpoint_analytic(np.array(x0_flat), tau)
    
    # Compare
    print(f"\nMidpoint (numerical): {x_mid_flat}")
    print(f"Midpoint (analytic) : {x_analytic}")
    print(f"Difference          : {np.abs(x_mid_flat - x_analytic)}")
    
    # Energy conservation (should be better than leapfrog)
    H0 = hamiltonian(x0)
    H_mid = hamiltonian(x_mid)
    H_analytic = hamiltonian(QP.from_array(jnp.array(x_analytic)))
    
    print(f"\nEnergy H(x0)        : {H0:.8f}")
    print(f"Energy H(x_mid)     : {H_mid:.8f}")
    print(f"Energy H(x_analytic): {H_analytic:.8f}")
    print(f"Energy error (IM)   : {np.abs(H_mid - H0):.2e}")
    
    # Assert accuracy
    assert np.allclose(x_mid_flat, x_analytic, atol=1e-5), "Implicit midpoint test failed!"
    print("\n✓ Implicit Midpoint test PASSED")


def test_energy_conservation():
    """Compare energy conservation between leapfrog and implicit midpoint"""
    print("\n" + "=" * 70)
    print("Energy Conservation Comparison (Multiple Steps)")
    print("=" * 70)
    
    # Setup
    dim = 1
    tau = 0.1
    N = 100  # Multiple steps
    mass_inv = jnp.eye(dim)
    target = gen_gaussian(dim =1)
    
    # Create Hamiltonian and gradient
    hamiltonian = gaussian_hamiltonian(mass_inv, target)
    gradH = jax.jit(jax.grad(hamiltonian))
    
    def gradH_qp(qp: QP) -> QP:
        grad_flat = gradH(qp)
        return grad_flat
    
    # Create integrators
    config_lf = IntegratorConfig(tau=tau, N=N)
    config_im = IntegratorConfig(tau=tau, N=N, tol=1e-8, max_iter=20)
    
    leapfrog = gen_leapfrog(gradH_qp, config_lf)
    midpoint = gen_midptFPI(gradH_qp, config_im)
    
    # Initial condition
    key = jax.random.PRNGKey(42)
    x0_flat = jax.random.normal(key, shape=(2 * dim,))
    x0 = QP.from_array(x0_flat)
    
    # Integrate
    x_lf = leapfrog(x0)
    x_mid = midpoint(x0)
    
    # Energy errors
    H0 = hamiltonian(x0)
    H_lf = hamiltonian(x_lf)
    H_mid = hamiltonian(x_mid)
    
    error_lf = np.abs(H_lf - H0)
    error_mid = np.abs(H_mid - H0)
    
    print(f"Initial energy H0   : {H0:.10f}")
    print(f"\nAfter {N} steps (tau={tau}):")
    print(f"Leapfrog    H_final : {H_lf:.10f}")
    print(f"Leapfrog    error   : {error_lf:.2e}")
    print(f"Midpoint    H_final : {H_mid:.10f}")
    print(f"Midpoint    error   : {error_mid:.2e}")
    print(f"\nMidpoint is {error_lf/error_mid:.1f}x more accurate")
    
    print("\n✓ Energy conservation test PASSED")


if __name__ == "__main__":
    # Check configuration
    print("JAX Configuration:")
    print(f"64-bit precision enabled: {jax.config.jax_enable_x64}")
    print()
    
    # Run tests
    test_leapfrog()
    test_implicit_midpoint()
    test_energy_conservation()
    
    print("\n" + "=" * 70)
    print("All tests PASSED! ✓")
    print("=" * 70)