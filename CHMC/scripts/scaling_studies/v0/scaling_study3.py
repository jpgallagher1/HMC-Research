import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import jax
import os
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

runs = 20  # divisible by 1, 2, 4, 10
num_devices = jax.device_count()
print(f"JAX devices: {num_devices}")

key = jax.random.PRNGKey(1)
# parameter sweep setup
dim = 2
dims = [2]
c_step = 11
runs = 20 #divisible by 1, 2, 4, 10
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

# reshape keys to (devices, runs_per_device, ...)
runs_per_device = runs // num_devices
hmc_keys_batch = jr.split(hmckey_runs, totalkeys).reshape(
    numτs, mcmc_iter_count, num_devices, runs_per_device, 2
)
chmc_keys_batch = jr.split(chmckey_runs, totalkeys).reshape(
    numτs, mcmc_iter_count, num_devices, runs_per_device, 2
)

run_idx = jnp.arange(runs).reshape(num_devices, runs_per_device)

# pmap over devices, vmap over runs_per_device within each device
def run_on_device(run_idx_device, hmc_keys_device, chmc_keys_device, n_samples):
    return jax.vmap(
        sample_one_run,
        in_axes=(None, 0, 0, 0, None, None, None)
    )(n_samples, run_idx_device, hmc_keys_device, chmc_keys_device, init_sample, H_flat, config)

pmapped_run = jax.pmap(run_on_device, in_axes=(0, 0, 0, None), static_broadcasted_argnums=(3,))


t_start = time.time()

for j, mainnum_samples in enumerate(lens):
    sample_hmc, sample_chmc = pmapped_run(
        run_idx,
        hmc_keys_batch[0, j],   # shape: (num_devices, runs_per_device, 2)
        chmc_keys_batch[0, j],
        int(mainnum_samples)
    )

    # flatten device dimension back to runs
    hmc_chainring = [
        extract_positions(samples, accepted_only=True)
        for samples in zip(*[s.reshape(-1, *s.shape[2:]) for s in sample_hmc])
    ]
    chmc_chainring = [
        extract_positions(samples, accepted_only=True)
        for samples in zip(*[s.reshape(-1, *s.shape[2:]) for s in sample_chmc])
    ]

    chmc_cov_matrices = jnp.array([metrics.cov(c) for c in chmc_chainring]).reshape(runs, dim, dim)
    hmc_cov_matrices = jnp.array([metrics.cov(c) for c in hmc_chainring]).reshape(runs, dim, dim)

    chmc_cov_metric[0, j, :] = vmaxtrdiff(chmc_cov_matrices, true_cov_matrices)
    hmc_cov_metric[0, j, :] = vmaxtrdiff(hmc_cov_matrices, true_cov_matrices)

t_end = time.time()
print(f"Sampling time: {t_end - t_start:.3f}s")