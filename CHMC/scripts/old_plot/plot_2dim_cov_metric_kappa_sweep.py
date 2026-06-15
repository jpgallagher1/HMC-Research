# John Gallagher
# May 20, 2026
# Script to show separation of kappa = 41:51


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import jax
from jax import jit
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from datatypes import QP, IntegratorConfig
from target import gen_2D_perturb_vals, gen_perturb_mat 
from hamiltonian import gaussian_hamiltonian
from sampler import hmc_sampler, chmc_sampler, extract_positions, sample_one_run
import metrics


jax.config.update("jax_enable_x64", True)
num_devices = jax.device_count()
print(f"JAX devices: {num_devices}")
# =============================================================================
# Configuration
# =============================================================================

dim = 2
runs = 20
tau = 0.2
N = 5
T = N * tau
tol = 1e-2
max_iter = 2

mcmc_min_exp = 2
mcmc_max_exp = 5
mcmc_iter_count = 9

key = jr.PRNGKey(1)


# =============================================================================
# Derived quantities
# =============================================================================

perturb_vals = gen_2D_perturb_vals(jnp.arange(41, 51))
num_perturbs = len(perturb_vals)

lens = jnp.logspace(mcmc_min_exp, mcmc_max_exp, mcmc_iter_count,
                    base=10, dtype=int)

config = IntegratorConfig(τ=tau, T=T, N=N, tol=tol, max_iter=max_iter)
Mass_inv = jnp.eye(dim)

# gen_perturb_mat returns the precision (Σ⁻¹) directly, which is what
# gaussian_hamiltonian expects. The covariance Σ = inv(precision) is used only
# as the reference matrix for the maxtracediff metric.
true_cov_mats = jax.vmap(gen_perturb_mat, in_axes=(None, 0))(
    dim, perturb_vals
)
target_mats = jax.vmap(jnp.linalg.inv)(true_cov_mats)


# =============================================================================
# PRNG key allocation
# =============================================================================
# One independent PRNGKey per (perturb, len, run) for each of HMC and CHMC,
# plus one init-key per perturb. Flat-index addressing avoids the collision
# trap of summing loop indices.

key_qp_inits, key_hmc, key_chmc = jr.split(key, 3)

qp_init_keys = jr.split(key_qp_inits, num_perturbs)             # (num_perturbs, 2)

total_chain_keys = num_perturbs * mcmc_iter_count * runs
hmc_keys = jr.split(key_hmc, total_chain_keys).reshape(
    num_perturbs, mcmc_iter_count, runs, 2
)
chmc_keys = jr.split(key_chmc, total_chain_keys).reshape(
    num_perturbs, mcmc_iter_count, runs, 2
)


# =============================================================================
# JIT-compiled samplers
# =============================================================================
# static_argnums covers H (callable) and config (NamedTuple with Python ints
# inside). solve for chmc is the default jnp.linalg.solve, also static.

jhmc_sampler = jit(hmc_sampler, static_argnums=(2, 3))
jchmc_sampler = jit(chmc_sampler, static_argnums=(2, 3, 4))


# =============================================================================
# Sweep
# =============================================================================
# Storage:
#   *_cov_metric: (num_perturbs, mcmc_iter_count, runs) — small, dense, plottable
#   *_chains:     dict keyed by (k, j, r) → (n_accepted, dim) jax array; ragged
#                 because n_accepted varies per chain. Saved to npz under
#                 distinct keys per chain.

hmc_cov_metric = np.zeros((num_perturbs, mcmc_iter_count, runs))
chmc_cov_metric = np.zeros((num_perturbs, mcmc_iter_count, runs))

hmc_chains = {}
chmc_chains = {}

t_start = time.time()

for k, val in enumerate(perturb_vals):
    H = gaussian_hamiltonian(target_mats[k], mass_inv=Mass_inv)
    H_flat = lambda qp_flat, H=H: H(QP.from_array(qp_flat))

    qp_init = jr.normal(qp_init_keys[k], shape=(2 * dim,))
    init_sample = [qp_init, 1, False]

    true_cov_k = true_cov_mats[k]

    for j, mainnum_samples in enumerate(lens):
        n_samples = int(mainnum_samples)

        for r in range(runs):
            # HMC chain
            hmc_keys_main = jr.split(hmc_keys[k, j, r], n_samples)
            sample_hmc = jhmc_sampler(init_sample, hmc_keys_main, H_flat, config)
            jax.block_until_ready(sample_hmc)

            # CHMC chain
            chmc_keys_main = jr.split(chmc_keys[k, j, r], n_samples)
            sample_chmc = jchmc_sampler(init_sample, chmc_keys_main, H_flat, config)
            jax.block_until_ready(sample_chmc)

            # Accepted-position chains (ragged length per chain).
            hmc_chain = extract_positions(sample_hmc, accepted_only=True)
            chmc_chain = extract_positions(sample_chmc, accepted_only=True)

            hmc_chains[(k, j, r)] = hmc_chain
            chmc_chains[(k, j, r)] = chmc_chain

            # Per-chain covariance error vs. the true Σ for this perturb.
            hmc_cov_metric[k, j, r] = metrics.maxtracediff(
                metrics.cov(hmc_chain), true_cov_k
            )
            chmc_cov_metric[k, j, r] = metrics.maxtracediff(
                metrics.cov(chmc_chain), true_cov_k
            )

t_end = time.time()
total_wall = t_end - t_start

print(f"\nSampling time: {total_wall:.3f}s")
######################
## k: perturb_vals
## j: lens
## r: runs
hmc_cov_metric.shape
mean_hmc_cov_metric = hmc_cov_metric.mean(axis=-1)
mean_chmc_cov_metric = chmc_cov_metric.mean(axis=-1)


conds = jax.vmap(lambda x: jnp.linalg.cond(x, np.inf))(true_cov_mats)

plt.figure(figsize=(14,10))
title = 'Average Cov Max Diag Err versus MCMC Iterations'
subtitle1 = f'\n$dim$ = {dims[0]}, $\\tau = $ {config.τ}, $N = $ {config.N}, κ$(A, L^\\infty)$'
for i in range(num_perturbs):
    chmc_p = np.polyfit(np.log(lens), np.log(mean_chmc_cov_metric[i]), 1)
    plt.loglog(lens, mean_chmc_cov_metric[i],'-*',  label =f'avg CHMC-κ={conds[i]: 0.1f}, slope: {chmc_p[0]: 0.2f}', alpha = 1)
    
    
for i in range(num_perturbs):
    hmc_p = np.polyfit(np.log(lens), np.log(mean_hmc_cov_metric[i]), 1)
    plt.loglog(lens, mean_hmc_cov_metric[i],'-o',  label =f'avg HMC-κ={conds[i]: 0.1f}, slope: {hmc_p[0]: 0.2f}', alpha = 1)
    
plt.title(title+subtitle1)
plt.xlabel('MCMC Iterations')
plt.ylabel('Error')
plt.grid(which='minor')
plt.grid(which='major')
plt.legend()


plt.savefig('plots/avg_cov_max_diag_err_scaling_vs_kappa41-50_oneplot.png')
print('plot saved')