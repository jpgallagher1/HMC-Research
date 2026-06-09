# John Gallagher
# May 27, 2026
# Script to show separation of kappa = 170:10:220
# Issues with original does not show convergence for 170, 190, 200:220 but expect some convergence for 170, 180, 190. This differs from previous by moving to new initial point for each run



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
from plotting import gen_τ_plots

import ot
import metrics

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('T', type = int)
parser.add_argument('k_min', type = int)
parser.add_argument('k_max', type = int)
parser.add_argument('k_step', type = int)
args = parser.parse_args()

jax.config.update("jax_enable_x64", True)
num_devices = jax.device_count()
print(f"JAX devices: {num_devices}")
# =============================================================================
# Configuration
# =============================================================================

dim = 2
dims = jnp.array([dim])
runs = 20
tau = 0.2
T = args.T
N = int(T/tau)
tol = 1e-2
max_iter = 2

mcmc_min_exp = 2
mcmc_max_exp = 5
mcmc_iter_count = 9

key = jr.PRNGKey(1)

# linspace
κ_min = args.k_min
κ_max = args.k_max
κ_steps = args.k_step

#setting up w1 metric
n_pts=1000
n_projections=100
mu_s = np.array([0, 0])



# =============================================================================
# Derived quantities
# =============================================================================
κs = jnp.linspace(κ_min, κ_max, κ_steps)
assert len(κs) == 9, 'must sweep 9 kappas'
perturb_vals = gen_2D_perturb_vals(κs)
num_perturbs = len(perturb_vals)



lens = jnp.logspace(mcmc_min_exp, mcmc_max_exp, mcmc_iter_count,
                    base=10, dtype=int)

config = IntegratorConfig(τ=tau, T=T, N=N, tol=tol, max_iter=max_iter)
Mass_inv = jnp.eye(dim)
print(f'Running: \ndim: {dim}, τ={config.τ}, N={config.N},  \n κs: {κs}, \n lens:{lens}')

# gen_perturb_mat returns the precision (Σ⁻¹) directly, which is what
# gaussian_hamiltonian expects. The covariance Σ = inv(precision) is used only
# as the reference matrix for the maxtracediff metric.
true_cov_mats = jax.vmap(gen_perturb_mat, in_axes=(None, 0))(
    dim, perturb_vals
)
target_mats = jax.vmap(jnp.linalg.inv)(true_cov_mats)

# for setup in wasserstein metric
a, b = np.ones((n_pts,)) / n_pts, np.ones((n_pts,)) / n_pts  # uniform distribution on samples

# =============================================================================
# PRNG key allocation
# =============================================================================
# One independent PRNGKey per (perturb, len, run) for each of HMC and CHMC,
# plus one init-key per perturb. Flat-index addressing avoids the collision
# trap of summing loop indices.

key_qp_inits, key_hmc, key_chmc = jr.split(key, 3)



total_chain_keys = num_perturbs * mcmc_iter_count * runs
qp_init_keys = jr.split(key_qp_inits, total_chain_keys).reshape(
    num_perturbs, mcmc_iter_count, runs, 2
)             # (num_perturbs, 2)
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
#   *_w1_metric: (num_perturbs, mcmc_iter_count, runs) — small, dense, plottable
#   *_chains:     dict keyed by (k, j, r) → (n_accepted, dim) jax array; ragged
#                 because n_accepted varies per chain. Saved to npz under
#                 distinct keys per chain.

hmc_w1_metric = np.zeros(shape=(num_perturbs, mcmc_iter_count, runs))
chmc_w1_metric = np.zeros(shape=(num_perturbs, mcmc_iter_count, runs))

hmc_chains = {}
chmc_chains = {}

t_start = time.time()

for k, val in enumerate(perturb_vals):
    H = gaussian_hamiltonian(target_mats[k], mass_inv=Mass_inv)
    H_flat = lambda qp_flat, H=H: H(QP.from_array(qp_flat))


    true_cov_k = true_cov_mats[k]

    for j, mainnum_samples in enumerate(lens):
        n_samples = int(mainnum_samples)
        print(f"{'#'*60}\n# dim: {dim}, tau: {config.τ}, N: {config.N}, "
              f"num_perturb_val: {k+1} of {num_perturbs}, len: {n_samples}")
        xt = ot.datasets.make_2D_samples_gauss(n_pts, mu_s, true_cov_k)
        for r in range(runs):
            qp_init = jr.normal(qp_init_keys[k,j,r], shape=(2 * dim,))
            init_sample = [qp_init, 1, False]
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

            hmc_chain_np = np.array(hmc_chain)
            chmc_chain_np = np.array(chmc_chain)
            a_hmc = np.ones(len(hmc_chain_np)) / len(hmc_chain_np)
            a_chmc = np.ones(len(chmc_chain_np)) / len(chmc_chain_np)
            # Per-chain covariance error vs. the true Σ for this perturb.
            try:
                hmc_w1_metric[k, j, r] = ot.sliced_wasserstein_distance(
                    hmc_chain_np, xt, a_hmc, b, n_projections, seed=j
                )
            except Exception:
                hmc_w1_metric[k, j, r] = 1
            try:
                chmc_w1_metric[k, j, r] = ot.sliced_wasserstein_distance(
                    chmc_chain_np, xt, a_chmc, b, n_projections, seed=j
                )
            except Exception:
                chmc_w1_metric[k, j, r] = 1

t_end = time.time()
total_wall = t_end - t_start

print(f"\nSampling time: {total_wall:.3f}s")
######################
## k: perturb_vals
## j: lens
## r: runs

mean_hmc_w1_metric = hmc_w1_metric.mean(axis=-1)
mean_chmc_w1_metric = chmc_w1_metric.mean(axis=-1)

conds = jax.vmap(lambda x: jnp.linalg.cond(x, np.inf))(true_cov_mats)

plt.figure(figsize=(14,10))
title = 'Average W1 distance versus MCMC Iterations'
subtitle1 = f'\n$dim$ = {dims[0]}, $\\tau = $ {config.τ}, $N = $ {config.N}, κ$(A, L^\\infty)$'
for i in range(num_perturbs):
    chmc_p = np.polyfit(np.log(lens), np.log(mean_chmc_w1_metric[i]), 1)
    plt.loglog(lens, mean_chmc_w1_metric[i],'-*',  label =f'avg CHMC-κ={conds[i]: 0.1f}, slope: {chmc_p[0]: 0.2f}', alpha = 1)

plt.gca().set_prop_cycle(None)    
    
for i in range(num_perturbs):
    hmc_p = np.polyfit(np.log(lens), np.log(mean_hmc_w1_metric[i]), 1)
    plt.loglog(lens, mean_hmc_w1_metric[i],'-o',  label =f'avg HMC-κ={conds[i]: 0.1f}, slope: {hmc_p[0]: 0.2f}', alpha = 1)
    
plt.title(title+subtitle1)
plt.xlabel('MCMC Iterations')
plt.ylabel('Error')
plt.grid(which='minor')
plt.grid(which='major')
plt.legend()


plt.savefig(f'plots/avg_w1_scaling_vs_kappa{κ_min}:{κ_max}_T{T}_oneplot2.png')
print('avg plot saved')
plt.close()

print('plotting grid now:')
fig, axes = plt.subplots(3, 3, figsize=(30, 30))
axes = axes.flatten()
for i, val in enumerate(conds):
    plt.sca(axes[i])
    gen_τ_plots(lens, chmc_w1_metric, hmc_w1_metric, config, i, dims=dims, cond= val, slope = True, avg=True, title=title)
plt.savefig(f'plots/avg_w1_scaling_vs_kappa{κ_min}:{κ_max}_T{T}_oneplot2_grid.png')
print('grid plot saved')
plt.close()