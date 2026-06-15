# John Gallagher
# May 14, 2026
# Script to replicated 2dim_plots_separation_test.ipynb covariance max diag error
#

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
from hamiltonian import gaussian_hamiltonian
from sampler import hmc_sampler, chmc_sampler, extract_positions
import metrics

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(1)

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

hmc_chainring = []
chmc_chainring = []
chainringindex = []
true_matrices = []

jhmc_sampler = jit(hmc_sampler, static_argnums=(2, 3))
jchmc_sampler = jit(chmc_sampler, static_argnums=(2, 3, 4))

Mass_inv = jnp.eye(dim)
key1, key2, key3 = jr.split(key, 3)
keyring = jr.split(key1, numτs)
chmckeyring = jr.split(key2, numτs * len(lens) * runs)
hmckeyring = jr.split(key3, numτs * len(lens) * runs)

high_κ_vec = jnp.array([101, -99])
κ100_mat = jnp.array([high_κ_vec, high_κ_vec[::-1]])

print('Starting for loop')
totaltime = time.time()
for k, val in enumerate(τs):
    N = 20
    T = val*N
    tol = 1e-2
    max_iter = 2
    config = IntegratorConfig(τ=val, 
                              T=T, 
                              N=N, 
                              tol=tol, 
                              max_iter=max_iter)
    target_mat = κ100_mat
    true_matrices.append(target_mat)
    H = gaussian_hamiltonian(target_mat, mass_inv=Mass_inv)
    H_flat = lambda qp_flat, H=H: H(QP.from_array(qp_flat))
    qp_init = jr.normal(keyring[k], shape=(2 * dim,))
    init_sample = [qp_init, 1, False]
    for j, mainnum_samples in enumerate(lens):
        for i in range(runs):
            # HMC samples
            hmc_keys_main = jr.split(hmckeyring[k + j + i], mainnum_samples)
            sample_hmc = jhmc_sampler(init_sample, hmc_keys_main, H_flat, config)
            jax.block_until_ready(sample_hmc)
            hmc_chainring.append(extract_positions(sample_hmc, accepted_only=True))
            
            # CHMC samples
            chmc_keys_main = jr.split(chmckeyring[k + j + i], mainnum_samples)
            sample_chmc = jchmc_sampler(init_sample, chmc_keys_main, H_flat, config)
            jax.block_until_ready(sample_chmc)
            chmc_chainring.append(extract_positions(sample_chmc, accepted_only=True))
            chainringindex.append((val, mainnum_samples))
print(f"Chain gen total time: {time.time() - totaltime:.3f}s")

chmc_cov_matrices = jnp.array([metrics.cov(c) for c in chmc_chainring]).reshape(numτs, mcmc_iter_count, runs, dim, dim)
hmc_cov_matrices = jnp.array([metrics.cov(c) for c in hmc_chainring]).reshape(numτs, mcmc_iter_count, runs, dim, dim)
true_matrices = jnp.array(true_matrices)

true_cov_matrices = jax.vmap(jnp.linalg.inv)(true_matrices)
vmaxtrdiff = jax.vmap(
    jax.vmap(jax.vmap(metrics.maxtracediff, in_axes=(0, None)), in_axes=(0, None)),
    in_axes=(0, 0),
)
chmc_cov_metric = vmaxtrdiff(chmc_cov_matrices, true_cov_matrices)
hmc_cov_metric = vmaxtrdiff(hmc_cov_matrices, true_cov_matrices)
conds = jax.vmap(lambda x: jnp.linalg.cond(x, p=np.inf))(true_cov_matrices)


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


gen_τ_plots(lens, chmc_cov_metric, hmc_cov_metric, config, 0, dims=dims, cond=conds[-3], slope=True)
plt.savefig('plots/verify/cov_max_diag_err2d_kappa100_diverge.png')
print('plot saved')
# plt.show()
