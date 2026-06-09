"""
Description:
    MCMC samplers: HMC and CHMC.
    USE THE CORRECT ENVIRONMENT:  HMC-Research

Author: John Gallagher
Created: 2026-02-16
Last Modified: 2026-02-16
Version: 0.1
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable, Tuple

from datatypes import QP, SamplerState, SamplerOutput, IntegratorConfig
from integrator import gen_leapfrog, gen_midptNewtonFPI

def gen_draw_momentum(c:float = 0.2, constant_p:bool =False):
    def draw_momentum_constant_p(qp: QP, key: jax.random.PRNGKey) -> Tuple[QP, None]:
        """
        Resample momentum from standard Gaussian.
        
        Keeps position q, resamples p ~ N(0, I)
        
        Args:
            qp: Current state
            key: JAX random key
            Only use constant_p if dim ==2
        Returns:
            (new_qp, None) - None for scan compatibility
        """
        _, drawkey = jr.split(key)
        p0 = jr.normal(drawkey)
        p1 = jnp.sqrt(c-p0**2)
        p_new = jnp.array([p0, p1])
        return QP(q=qp.q, p = p_new), None
   
    def draw_momentum(qp: QP, key: jax.random.PRNGKey) -> Tuple[QP, None]:
        """
        Resample momentum from standard Gaussian.
        
        Keeps position q, resamples p ~ N(0, I)
        
        Args:
            qp: Current state
            key: JAX random key
        Returns:
            (new_qp, None) - None for scan compatibility
        """
        _, drawkey = jr.split(key)
        p_new = jr.normal(drawkey, shape=qp.q.shape)
        return QP(q=qp.q, p=p_new), None
    if constant_p:
        return draw_momentum_constant_p
    else:
        return draw_momentum


def draw_momentum(qp: QP, key: jax.random.PRNGKey) -> Tuple[QP, None]:
    """
    Resample momentum from standard Gaussian.
    
    Keeps position q, resamples p ~ N(0, I)
    
    Args:
        qp: Current state
        key: JAX random key
    Returns:
        (new_qp, None) - None for scan compatibility
    """
    _, drawkey = jr.split(key)
    p_new = jr.normal(drawkey, shape=qp.q.shape)
    return QP(q=qp.q, p=p_new), None

def accept_reject(delta_H: float, key: jax.random.PRNGKey) -> bool:
    """
    Metropolis-Hastings accept/reject step.
    
    Accept probability: min(1, exp(delta_H))
    
    Args:
        delta_H: Energy difference (H_proposed - H_current)
        key: Random key
        
    Returns:
        True if accepted, False otherwise
    """
    _, drawkey = jr.split(key)
    alpha = jnp.minimum(1.0, jnp.exp(delta_H))
    u = jr.uniform(drawkey, shape=())
    return u <= alpha

def gen_hmc_kernel(
    H: Callable[[jnp.ndarray], float],
    config: IntegratorConfig
) -> Callable:
    """
    Generate HMC kernel using leapfrog integrator.
    
    Args:
        H: Hamiltonian function (takes flat array)
        tau: Integration step size
        N: Number of integration steps
        
    Returns:
        HMC kernel function
    """
    gradH = jax.grad(H)
    integrator = gen_leapfrog(gradH, config)
    draw_momentum = gen_draw_momentum(constant_p=config.constant_p)
    
    def hmc_kernel(carry_in, key):
        """
        Single HMC step.
        
        Args:
            carry_in: [qp_flat, delta_H, accepted] (previous state)
            key: Random key
            
        Returns:
            (carry_out, carry_out) for scan
        """
        qp_flat, _, _ = carry_in
        qp = QP.from_array(qp_flat)
        
        # Resample momentum
        qp0, _ = draw_momentum(qp, key)
        qp0_flat = qp0.to_array()
        
        # Integrate
        qp_star_flat = integrator(qp0_flat)
        
        # Accept/reject
        delta_H = H(qp0_flat) - H(qp_star_flat)  # -(final - init)
        is_accepted = accept_reject(delta_H, key)
        
        qp_out_flat = jnp.where(is_accepted, qp_star_flat, qp0_flat)
        carry_out = [qp_out_flat, delta_H, is_accepted]
        
        return carry_out, carry_out
    
    return hmc_kernel

def gen_chmc_kernel(
    H: Callable[[jnp.ndarray], float],
    config: IntegratorConfig,
    solve: Callable = jnp.linalg.solve
) -> Callable:
    """
    Generate CHMC kernel using implicit midpoint FPI.
    
    Args:
        H: Hamiltonian function (takes flat array)
        tau: Integration step size
        N: Number of integration steps
        tol: FPI convergence tolerance
        max_iter: Maximum Newton iterations per step
        solve: Linear solver for Newton's method
        
    Returns:
        CHMC kernel function
    """
    gradH = jax.grad(H)
    integrator = gen_midptNewtonFPI(gradH, config, solve)
    
    def chmc_kernel(carry_in, key):
        """
        Single CHMC step.
        
        Args:
            carry_in: [qp_flat, delta_H, accepted]
            key: Random key
            
        Returns:
            (carry_out, carry_out) for scan
        """
        qp_flat, _, _ = carry_in
        qp = QP.from_array(qp_flat)
        
        # Resample momentum
        qp0, _ = draw_momentum(qp, key)
        qp0_flat = qp0.to_array()
        
        # Integrate with implicit method
        qp_star_flat = integrator(qp0_flat)
        
        # Accept/reject
        delta_H = H(qp0_flat) - H(qp_star_flat)
        is_accepted = accept_reject(delta_H, key)
        
        qp_out_flat = jnp.where(is_accepted, qp_star_flat, qp0_flat)
        carry_out = [qp_out_flat, delta_H, is_accepted]
        
        return carry_out, carry_out
    
    return chmc_kernel

def hmc_sampler(
    initial_sample: list,
    keys: jax.random.PRNGKey,
    H: Callable,
    config: IntegratorConfig
) -> Tuple:
    """
    Run HMC sampler.
    
    Args:
        initial_sample: [qp_flat, delta_H, accepted]
        keys: Array of random keys (one per sample)
        H: Hamiltonian function
        tau: Step size
        N: Number of integration steps
        
    Returns:
        samples: [qp, delta_H, accepted] for each iteration
    """
    hmc_kernel = gen_hmc_kernel(H, config)
    _, samples = jax.lax.scan(hmc_kernel, initial_sample, xs=keys)
    return samples
def chmc_sampler(
    initial_sample: list,
    keys: jax.random.PRNGKey,
    H: Callable,
    config: IntegratorConfig,
    solve: Callable = jnp.linalg.solve
) -> Tuple:
    """
    Run CHMC sampler.
    
    Args:
        initial_sample: [qp_flat, delta_H, accepted]
        keys: Array of random keys
        H: Hamiltonian function
        tau: Step size
        N: Number of integration steps
        tol: FPI tolerance
        max_iter: Max Newton iterations
        solve: Linear solver
        
    Returns:
        samples: [qp, delta_H, accepted] for each iteration
    """
    chmc_kernel = gen_chmc_kernel(H, config, solve)
    _, samples = jax.lax.scan(chmc_kernel, initial_sample, xs=keys)
    return samples

def sample_one_run(mainnum_samples, run_idx, hmc_key, chmc_key, init_sample, H_flat, config):
    """Sample one run - returns raw samples
    sample_hmc, sample_chmc
    """
    hmc_keys_main = jr.split(hmc_key, mainnum_samples)
    sample_hmc = hmc_sampler(init_sample, hmc_keys_main, H_flat, config)
    
    chmc_keys_main = jr.split(chmc_key, mainnum_samples)
    sample_chmc = chmc_sampler(init_sample, chmc_keys_main, H_flat, config)
    
    return sample_hmc, sample_chmc

def extract_positions(samples: Tuple, accepted_only: bool = False) -> jnp.ndarray:
    """
    Extract position samples from sampler output.
    
    Args:
        samples: Output from hmc_sampler or chmc_sampler
        
    Returns:
        (n_samples, dim) array of positions
    """
    qp_samples, _, accepted = samples
    # Each qp_sample is a flat array [q, p]
    dim = qp_samples.shape[1] // 2
    if accepted_only:
        return qp_samples[:, :dim][accepted]
    else: 
        return qp_samples[:, :dim]

def extract_energy(samples: Tuple, accepted_only: bool = False) -> jnp.ndarray:
    """
    Extract position samples from sampler output.
    
    Args:
        samples: Output from hmc_sampler or chmc_sampler
        
    Returns:
        (n_samples, dim) array of positions
    """
    _, ΔH, accepted = samples
    # Each qp_sample is a flat array [q, p]
    if accepted_only:
        return ΔH[accepted]
    else: 
        return ΔH

