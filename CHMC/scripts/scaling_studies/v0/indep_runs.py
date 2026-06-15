# John Gallagher
# May 18, 2026
# Script to deploy single instance of hmc to pinnacles based on given slurm script for array of jobs
#


from pathlib import Path
import sys

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

import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('run_id', type=int)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

# Access command-line arguments

inkey = args.run_id


jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(inkey)

# Integrator configuration
dim = 2
dims = [2]
runs = 1
τs = np.array([0.2])
numτs = len(τs)
lens = int(1000)
N = 20
T = τs[0]*N
tol = 1e-2
max_iter = 2
config = IntegratorConfig(τ=τs[0], T=T, N=N, tol=tol, max_iter=max_iter)



# Setting up rng keys
key_init, key2, key3 = jr.split(key, 3)
hmc_keys_main = jr.split(key2, lens)
chmc_keys_main = jr.split(key3, lens)

# Setting up Hamiltonian
high_κ_vec = jnp.array([101, -99])
κ100_mat = jnp.array([high_κ_vec, high_κ_vec[::-1]])
target_mat = κ100_mat
Mass_inv = jnp.eye(dim)
H = gaussian_hamiltonian(target_mat, mass_inv=Mass_inv)
H_flat = lambda qp_flat, H=H: H(QP.from_array(qp_flat))
qp_init = jr.normal(key_init, shape=(2 * dim,))
init_sample = [qp_init, 1, False]

# jit sampler
jhmc_sampler = jit(hmc_sampler, static_argnums=(2, 3))
jchmc_sampler = jit(chmc_sampler, static_argnums=(2, 3, 4))


sample_hmc = jhmc_sampler(init_sample, hmc_keys_main, H_flat, config)
jax.block_until_ready(sample_hmc)



sample_chmc = jchmc_sampler(init_sample, chmc_keys_main, H_flat, config)
jax.block_until_ready(sample_chmc)


totaltime = time.time()

chmc_cov_matrices = metrics.cov(extract_positions(sample_chmc, accepted_only=True))
hmc_cov_matrices = metrics.cov(extract_positions(sample_hmc, accepted_only=True))

target_cov_matrices = jnp.linalg.inv(target_mat)

chmc_cov_metric = metrics.maxtracediff(chmc_cov_matrices, target_cov_matrices)
hmc_cov_metric = metrics.maxtracediff(hmc_cov_matrices, target_cov_matrices)
conds = jnp.linalg.cond(target_cov_matrices, p=np.inf)

# Save results to file
results = {
    'chmc_cov_metric': float(chmc_cov_metric),
    'hmc_cov_metric': float(hmc_cov_metric),
    # 'chmc_positions': np.array(extract_positions(sample_chmc, accepted_only=True)),
    # 'hmc_positions': np.array(extract_positions(sample_hmc, accepted_only=True))
}

np.save(args.output, results)
print(f'Run {inkey} | chmc_cov_metric: {chmc_cov_metric}, hmc_cov_metric: {hmc_cov_metric} | Saved to {args.output}')