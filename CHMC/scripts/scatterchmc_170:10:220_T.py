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
# from scipy.ndimage.filters import gaussian_filter

import ot
import metrics

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('T', type = int)
args = parser.parse_args()

jax.config.update("jax_enable_x64", True)
num_devices = jax.device_count()
print(f"JAX devices: {num_devices}")
# =============================================================================
# Configuration
# =============================================================================

dim = 2
dims = jnp.array([dim])
runs = 1
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
κ_min = 170
κ_max = 220
κ_steps = 6

#setting up w1 metric
n_pts=1000
n_projections=100
mu_s = np.array([0, 0])


# =============================================================================
# Derived quantities
# =============================================================================
κs = jnp.linspace(κ_min, κ_max, κ_steps)
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

chmc_keys = jr.split(key_chmc, total_chain_keys).reshape(
    num_perturbs, mcmc_iter_count, runs, 2
)


# =============================================================================
# JIT-compiled samplers
# =============================================================================
# static_argnums covers H (callable) and config (NamedTuple with Python ints
# inside). solve for chmc is the default jnp.linalg.solve, also static.

jchmc_sampler = jit(chmc_sampler, static_argnums=(2, 3, 4))


# =============================================================================
# Sweep
# =============================================================================
# Storage:
#   *_w1_metric: (num_perturbs, mcmc_iter_count, runs) — small, dense, plottable
#   *_chains:     dict keyed by (k, j, r) → (n_accepted, dim) jax array; ragged
#                 because n_accepted varies per chain. Saved to npz under
#                 distinct keys per chain.


chmc_chains = {}

t_start = time.time()

for k, val in enumerate(perturb_vals):
    H = gaussian_hamiltonian(target_mats[k], mass_inv=Mass_inv)
    H_flat = lambda qp_flat, H=H: H(QP.from_array(qp_flat))


    true_cov_k = true_cov_mats[k]
    print(f"{'#'*60}\n# dim: {dim}, tau: {config.τ}, T: {config.T}, N: {config.N}, "
            f"num_perturb_val: {k+1} of {num_perturbs}"
            )

    for j, mainnum_samples in enumerate(lens):
        n_samples = int(mainnum_samples)

        for r in range(runs):
            qp_init = jr.normal(qp_init_keys[k,j,r], shape=(2 * dim,))
            init_sample = [qp_init, 1, False]
            
            # CHMC chain
            chmc_keys_main = jr.split(chmc_keys[k, j, r], n_samples)
            sample_chmc = jchmc_sampler(init_sample, chmc_keys_main, H_flat, config)
            jax.block_until_ready(sample_chmc)

            # Accepted-position chains (ragged length per chain).
            chmc_chain = extract_positions(sample_chmc, accepted_only=True)
            chmc_chains[(k, j, r)] = chmc_chain


t_end = time.time()
total_wall = t_end - t_start

print(f"\nSampling time: {total_wall:.3f}s")
######################
## k: perturb_vals
## j: lens
## r: runs

print('plotting grid now:')
fig, axes = plt.subplots(2, 3, figsize=(30, 20))
axes = axes.flatten()
for k, val in enumerate(conds):
    chmc_samples = min(2000, chmc_chains[(k,8,0)].shape[0])
    try: 
        xchmc, ychmc = chmc_chains[(k,8,0)][:max_chmc_samples,:].T
    except:
        xchmc, ychmc = [],[]
    
    heatmapchmc, xedgeschmc, yedgeschmc = np.histogram2d(xchmc, ychmc, bins=90)
    extent = [xedgeschmc[0],xedgeschmc[-1], yedgeschmc[0], yedgeschmc[-1]]

    xvals = np.linspace(extent[0], extent[1], 3001)
    yvals = np.linspace(extent[2], extent[3], 3001)
    X, Y = jnp.meshgrid(xvals, yvals)
    XY = jnp.stack([X, Y], axis=-1)
    print(XY.shape)
    AXY = XY @ target_mats[0].T
    print(AXY.shape)
    Z = jnp.einsum('...i,...i ->...', XY,AXY)
    pdf = np.exp(-0.5 * Z)
    levels = np.linspace(pdf.min(), pdf.max(), 4)

    plt.clf()
    if xchmc == []:
        chmc_label = 'No Samples'
    else:
        chmc_label = 'CHMC Samples'
    plt.scatter(xchmc,ychmc, marker='+',c='orange', alpha=0.05, label=chmc_label)
    plt.legend().legend_handles[0].set_alpha(1)

    plt.contour(X,Y, pdf, cmap='plasma', levels = levels, alpha = 0.5)
    plt.imshow(heatmapchmc, extent=extent, origin='lower')
    
    subtitle1 = f'\n$dim$ = {dims[0]}, $\\tau = $ {config.τ}, $N = $ {config.N}, κ={conds[k]}'
    plt.title('HMC/CHMC Contour and Histogram plot\n2_000 Samples'+subtitle1)
    plt.legend().set_loc('upper left')

    plt.show()

figpath2 = f'plots/scatterCHMC{κ_min}:{κ_max}_T{T}_oneplot2_grid.png'
plt.savefig(figpath2)
print('grid plot saved',f'\n{figpath2}')
plt.close()