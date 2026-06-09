# John Gallagher
# May 18, 2026
# scaling study 2 to see how this code scales on a few cores. 
# Script to replicated 2dim_plots_separation_test.ipynb covariance max diag error
# updated to have vmap over inner nested 20 runs. 

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import jax

print(f"JAX devices: {jax.device_count('cpu')}")

from jax import jit
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from datatypes import QP, IntegratorConfig
from hamiltonian import gaussian_hamiltonian
from sampler import hmc_sampler, chmc_sampler, extract_positions, sample_one_run
import metrics

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(1)

# parameter sweep setup
dim = 2
dims = [2]
c_step = 11
runs = 20
τs = np.array([0.2])
numτs = len(τs)
mcmc_min_exp = 2
mcmc_max_exp = 5
mcmc_iter_count = 9
lens = jnp.logspace(mcmc_min_exp, mcmc_max_exp, mcmc_iter_count, base=10, dtype=int)

# initial rng key setup, one to seed each run
totalkeys = numτs*mcmc_iter_count*runs # 2 b/c hmc & chmc

key_init, hmckey_runs, chmckey_runs = jr.split(key, 3)
hmc_keys_batch = jr.split(hmckey_runs, totalkeys).reshape(numτs,mcmc_iter_count,runs,2) 
chmc_keys_batch = jr.split(chmckey_runs, totalkeys).reshape(numτs,mcmc_iter_count,runs,2)

##
# Hamiltonian Setup
##
# target

high_κ_vec = jnp.array([101, -99])
κ100_mat = jnp.array([high_κ_vec, high_κ_vec[::-1]])
target_mat = κ100_mat
Mass_inv = jnp.eye(dim)

H = gaussian_hamiltonian(target_mat, mass_inv=Mass_inv)
H_flat = lambda qp_flat, H=H: H(QP.from_array(qp_flat))
qp_init = jr.normal(key_init, shape=(2 * dim,))
init_sample = [qp_init, 1, False]

##
# HMC & CHMC Integrator Setup
##

N = 20
T = τs[0]*N
tol = 1e-2
max_iter = 2
config = IntegratorConfig(τ=τs[0], 
                            T=T, 
                            N=N, 
                            tol=tol, 
                            max_iter=max_iter)

jhmc_sampler = jit(hmc_sampler, static_argnums=(2, 3))
jchmc_sampler = jit(chmc_sampler, static_argnums=(2, 3, 4))


vmapped_run = jax.vmap(
    sample_one_run, 
    in_axes=(None, 0, 0, 0, None, None, None)
)

# vmapping metric

true_cov_matrices = jnp.linalg.inv(κ100_mat)
conds = jnp.linalg.cond(true_cov_matrices,p=np.inf)

vmaxtrdiff = jax.vmap(metrics.maxtracediff, in_axes=(0, None))




run_idx = jnp.arange(runs)

hmc_cov_metric = np.zeros(shape=(numτs, mcmc_iter_count, runs))
chmc_cov_metric = np.zeros(shape=(numτs, mcmc_iter_count, runs))

for j, mainnum_samples in enumerate(lens):
        sample_hmc, sample_chmc = vmapped_run(mainnum_samples,
            run_idx, 
            hmc_keys_batch[0,j,:],
            chmc_keys_batch[0,j,:],
            init_sample,
            H_flat,
            config
        )
        # Extract positions (post-processing)
        hmc_chainring = [
            extract_positions(samples, accepted_only=True)
            for samples in zip(*sample_hmc)
        ]

        chmc_chainring = [
            extract_positions(samples, accepted_only=True)
            for samples in zip(*sample_chmc)
        ]


        chmc_cov_matrices = jnp.array([metrics.cov(c) for c in chmc_chainring]).reshape(runs, dim, dim)
        hmc_cov_matrices = jnp.array([metrics.cov(c) for c in hmc_chainring]).reshape(runs, dim, dim)

        chmc_cov_metric[0,j,:] = vmaxtrdiff(chmc_cov_matrices, true_cov_matrices)
        hmc_cov_metric[0,j,:] = vmaxtrdiff(hmc_cov_matrices, true_cov_matrices)



def gen_τ_plots(
    lens,
    chmc_metrics,
    hmc_metrics,
    config,
    i,
    dims=dims,
    cond=conds,
    slope=False,
    avg=True,
    title="Cov Max Diag Err versus MCMC Iterations",
):
    plt.loglog(lens, chmc_metrics[i, :], "-*", color="C0", alpha=0.15)
    plt.loglog(lens, hmc_metrics[i, :], "-o", color="C1", alpha=0.15)
    subtitle1 = f"\n$dim$ = {dims[0]}, $\\kappa$ = {cond: 0.2f}, $\\tau = $ {config.τ}, $N = $ {config.N}"
    if avg:
        avg_chmc = np.mean(chmc_metrics[i, :], axis=1)
        plt.loglog(lens, avg_chmc, "-*", color="C0", label="avg CHMC", alpha=1)
        avg_hmc = np.mean(hmc_metrics[i, :], axis=1)
        plt.loglog(lens, avg_hmc, "-o", color="C1", label="avg HMC", alpha=1)
    if slope:
        chmc_p = np.polyfit(np.log(lens), np.log(avg_chmc), 1)
        hmc_p = np.polyfit(np.log(lens), np.log(avg_hmc), 1)
        subtitle2 = f"\n CHMC avg. slope = {chmc_p[0]:.2f}, HMC avg. slope = {hmc_p[0]:.2f}"
        plt.title(title + subtitle1 + subtitle2)
    else:
        plt.title(title + subtitle1)
    plt.xlabel("MCMC Iterations")
    plt.ylabel("Error")
    plt.grid(which="minor")
    plt.grid(which="major")
    plt.legend()


gen_τ_plots(lens, chmc_cov_metric, hmc_cov_metric, config, 0, dims=dims, cond=conds, slope=True)
# plt.close()
plt.savefig('/home/johngallagher/data/HMC-Research/plots/scaling_study.png')
# print('plot saved')
# plt.show()