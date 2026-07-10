# John Gallagher
# July 8, 2026
# Script to test pgeneralized gauss in McGregor & Wan '26
# 



import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


import jax
from jax import jit
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from datatypes import QP, IntegratorConfig
from target import gen_2D_perturb_vals, gen_perturb_mat, gen_p_gauss_pdf
from hamiltonian import gaussian_hamiltonian, standard_hamiltonian, p_gauss_hamiltonian
from sampler import hmc_sampler, chmc_sampler, extract_positions, sample_one_run
from plotting import gen_τ_plots
import metrics

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('dmin', type = int)
parser.add_argument('dmax', type = int)
parser.add_argument('dsteps', type = int)
args = parser.parse_args()

# tau_str = f'{args.tau:.1f}'.replace('.', '_')

jax.config.update("jax_enable_x64", True)
num_devices = jax.device_count()
print(f"JAX devices: {num_devices}")
# =============================================================================
# Configuration
# =============================================================================

# logspace
d_min = args.dmin
d_max = args.dmax
d_steps = args.dsteps
# dims = jnp.array([1280, 2560, 5120, 10240, 20480, 40960])
#                   ([2^7, 2^8, 2^9, 2^10, 2^11, 2^12])*10                    
# dims = jnp.array([20, 40, 60, 80, 160, 320, 640]) #  Smaller dimensions for testing

β = 4

runs = 10
tau = 0.15
N = 26
T = N*tau
tol = 1e-2
max_iter = 2

# mcmc_min_exp = 2
# mcmc_max_exp = 3
mcmc_iter_count = 11

lens = jnp.linspace(100,5000,mcmc_iter_count)

key = jr.PRNGKey(1)



# =============================================================================
# Derived quantities
# =============================================================================

dims = jnp.logspace(d_min, d_max, d_steps, base=2, dtype=int)*10

num_dims = len(dims)
dim = dims[0]


# lens = jnp.logspace(mcmc_min_exp, mcmc_max_exp, mcmc_iter_count,
#                     base=10, dtype=int)


config = IntegratorConfig(τ=tau, T=T, N=N, tol=tol, max_iter=max_iter, integrator='AVF_NewtonFPI_T', gen_gauss=True)
Mass_inv = None
H = p_gauss_hamiltonian(4, None)

# Not static shapes so can't vmap
true_cov_mats_diag = []
for i,d in enumerate(dims): 
    true_cov_mats_diag.append(jnp.ones(d))
    print(f'Running: \ndim: {d}, τ={config.τ}, N={config.N},  \n lens:{lens}, p-gauss: β={β}')


# gen_perturb_mat returns the precision (Σ⁻¹) directly, which is what
# gaussian_hamiltonian expects. The covariance Σ = inv(precision) is used only
# as the reference matrix for the maxtracediff metric.


# =============================================================================
# PRNG key allocation
# =============================================================================
# One independent PRNGKey per (perturb, len, run) for each of HMC and CHMC,
# plus one init-key per perturb. Flat-index addressing avoids the collision
# trap of summing loop indices.

key_qp_inits, key_hmc, key_chmc = jr.split(key, 3)



total_chain_keys = num_dims * mcmc_iter_count * runs
qp_init_keys = jr.split(key_qp_inits, total_chain_keys).reshape(
    num_dims, mcmc_iter_count, runs, 2
)             # (num_dims, 2)
hmc_keys = jr.split(key_hmc, total_chain_keys).reshape(
    num_dims, mcmc_iter_count, runs, 2
)
chmc_keys = jr.split(key_chmc, total_chain_keys).reshape(
    num_dims, mcmc_iter_count, runs, 2
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
#   *_cov_metric: (num_dims, mcmc_iter_count, runs) — small, dense, plottable
#   *_chains:     dict keyed by (k, j, r) → (n_accepted, dim) jax array; ragged
#                 because n_accepted varies per chain. Saved to npz under
#                 distinct keys per chain.

hmc_cov_metric = np.zeros((num_dims, mcmc_iter_count, runs))
chmc_cov_metric = np.zeros((num_dims, mcmc_iter_count, runs))

# hmc_chains = {} #can't use for large dim runs
# chmc_chains = {} #can't use for large dim runs

t_start = time.time()

for k, val in enumerate(dims):
    # target = gen_p_gauss_pdf(β)
    
    true_cov_diag_k = true_cov_mats_diag[k]
    print(f"{'#'*60}\n# dim: {val}, num_dim: {k+1} of {num_dims}", f"tau: {config.τ}, T: {config.T}, N: {config.N}")


    for j, mainnum_samples in enumerate(lens):
        n_samples = int(mainnum_samples)

        for r in range(runs):
            qp_init = QP(jr.normal(qp_init_keys[k,j,r], shape=(2 * val,)))
            init_sample = [qp_init, 1, False]
            # HMC chain
            hmc_keys_main = jr.split(hmc_keys[k, j, r], n_samples)
            sample_hmc = jhmc_sampler(init_sample, hmc_keys_main, H, config)
            jax.block_until_ready(sample_hmc)

            # CHMC chain
            chmc_keys_main = jr.split(chmc_keys[k, j, r], n_samples)
            sample_chmc = jchmc_sampler(init_sample, chmc_keys_main, H, config)
            jax.block_until_ready(sample_chmc)


            # Accepted-position chains (ragged length per chain).
            hmc_chain = extract_positions(sample_hmc, accepted_only=True)
            chmc_chain = extract_positions(sample_chmc, accepted_only=True)

            # hmc_chains[(k, j, r)] = hmc_chain #can't use for large dim runs
            # chmc_chains[(k, j, r)] = chmc_chain #can't use for large dim runs

            # Per-chain covariance error vs. the true Σ for this perturb.
            hmc_cov_metric[k, j, r] = metrics.maxdiff(
                metrics.cov_diag(hmc_chain), true_cov_diag_k
            )
            chmc_cov_metric[k, j, r] = metrics.maxdiff(
                metrics.cov_diag(chmc_chain), true_cov_diag_k
            )

t_end = time.time()
total_wall = t_end - t_start
print(f"\nSampling time: {total_wall:.3f}s")


######################
## k: perturb_vals
## j: lens
## r: runs

mean_hmc_cov_metric = hmc_cov_metric.mean(axis=-1)
mean_chmc_cov_metric = chmc_cov_metric.mean(axis=-1)

# conds = jax.vmap(lambda x: jnp.linalg.cond(x, np.inf))(true_cov_mats_diag)

plt.figure(figsize=(14,10))
title = 'Average Cov Max Diag Err versus MCMC Iterations'
subtitle1 = f'\n $\\tau = $ {config.τ}, $N = $ {config.N}, Gen-Gauss: $\\beta = 4$'
for i in range(num_dims):
    chmc_p = np.polyfit(np.log(lens), np.log(mean_chmc_cov_metric[i]), 1)
    plt.loglog(lens, mean_chmc_cov_metric[i],'-*',  label =f'avg CHMC-d:{dims[i]}, slope: {chmc_p[0]: 0.2f}', alpha = 1)

plt.gca().set_prop_cycle(None)    
    
for i in range(num_dims):
    hmc_p = np.polyfit(np.log(lens), np.log(mean_hmc_cov_metric[i]), 1)
    plt.loglog(lens, mean_hmc_cov_metric[i],'-o',  label =f'avg HMC-d:{dims[i]}, slope: {hmc_p[0]: 0.2f}', alpha = 1)
    
plt.title(title+subtitle1)
plt.xlabel('MCMC Iterations')
plt.ylabel('Error')
plt.grid(which='minor')
plt.grid(which='major')
plt.legend()

figpath1 = f'plots/gen_gauss_avg_cov_max_diag_err_scaling_vs_dim{d_min}:{d_max}_τ0_15_oneplot.png'
plt.savefig(figpath1)
print('avg plot saved',f'\n{figpath1}')
plt.close()
# plt.show()


print('plotting grid now:')
fig, axes = plt.subplots(2, 3, figsize=(50, 20))
axes = axes.flatten()
for i, val in enumerate(dims):
    plt.sca(axes[i])
    gen_τ_plots(lens, chmc_cov_metric, hmc_cov_metric, config, i, dims=dims, cond= val,kappa=False, dim = True, slope = True, avg=True)
    
figpath2 = f'plots/gen_gauss_avg_cov_max_diag_err_scaling_vs_dim{d_min}:{d_max}_τ0_15_one_plot_grid.png'
plt.savefig(figpath2)
print('grid plot saved',f'\n{figpath2}')
plt.close()
