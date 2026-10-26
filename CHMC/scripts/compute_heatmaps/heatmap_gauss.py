# October 20, 2026
# Compute-time heatmap data on the Gaussian target with covariance [[1, rho], [rho, 1]],
# rho = (kappa - 1)/(kappa + 1). One chain of `length` samples per run; the wall time of
# each chain is saved (compilation is excluded). Same layout as heatmap.py:
#   METHOD/tau_*/T_*/tol_*/len_*/run_*.npz       (LF also sits under tol_*, as in heatmap.ipynb;
#                                                 its tol is unused, so its tol columns are repeats)
# Existing run files are skipped, so a rerun resumes. Plots: heatmap_gauss_plots.py.

import sys
from pathlib import Path
import argparse
import json
import os
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
jax.config.update("jax_enable_x64", True)
import jax.random as jr
import jax.numpy as jnp
import numpy as np

from datatypes import QP, gen_configs
from hamiltonian import gaussian_hamiltonian
from sampler import gen_chmc_kernel, gen_hmc_kernel
from target import gen_perturb_mat

# =============================================================================
# Configuration
# =============================================================================

dim = 2
kappa = 256
rho = (kappa - 1)/(1 + kappa)
Mass_inv = jnp.eye(dim)

n_runs = 10
methods = ['LF', 'AA', 'Newt', 'FPI']
integrators = ['LF', 'AA', 'AVF_NewtonFPI_T', 'AVF_FPI_T']
taus = [0.0625, 0.1, 0.125, 0.25, 0.5]
T = 3.0
tols = [1e-3, 1e-2, 1e-1, 3e-1]
length = 10_000
max_iter = 10                  # K cap for Newton; iteration cap for AA and FPI
m = 3
n_pts = 6

# Optional overrides for short runs; experiment settings live above.
parser = argparse.ArgumentParser()
parser.add_argument('--kappa', type=float, default=kappa)
parser.add_argument('--output', type=Path, default=None)
parser.add_argument('--runs', type=int, default=n_runs)
parser.add_argument('--length', type=int, default=length)
parser.add_argument('--taus', type=float, nargs='+', default=taus)
parser.add_argument('--tols', type=float, nargs='+', default=tols)
args = parser.parse_args()
kappa = int(args.kappa) if float(args.kappa).is_integer() else args.kappa
rho = (kappa - 1)/(1 + kappa)
n_runs, length, taus, tols = args.runs, args.length, args.taus, args.tols
base = args.output or Path(
    f'/Users/johngallagher/Documents/UCMerced/Research/GitHub/HMC-Online/results/heatmap_k{kappa}_covmat')

assert n_runs > 0 and length > 0 and max_iter > 0 and m > 0 and n_pts > 0
assert all(np.isfinite(x) and x > 0 for x in [*taus, T, *tols])

# =============================================================================
# Derived quantities and metadata
# =============================================================================

true_cov = gen_perturb_mat(dim, rho)  # [[1, rho], [rho, 1]] is the COVARIANCE
precision = jnp.linalg.inv(true_cov)   # the Hamiltonian takes the precision
H = gaussian_hamiltonian(precision, mass_inv=Mass_inv)

print(f'JAX devices: {jax.devices()}')
print(f'dim: {dim}, kappa: {kappa}, rho: {rho}, T: {T}, length: {length}')

base.mkdir(parents=True, exist_ok=True)
parameters = dict(
    schema=1, dim=dim, kappa=kappa, rho=rho, matrix_role='covariance',
    methods=methods, integrators=integrators, taus=taus, T=T, tols=tols,
    length=length, max_iter=max_iter, AA_m=m, n_pts=n_pts, n_runs=n_runs,
    jax_version=jax.__version__, backend=jax.default_backend(),
)
metadata_path = base / 'metadata.json'
if metadata_path.exists():
    if json.loads(metadata_path.read_text()) != parameters:
        raise ValueError('Saved settings differ. Use a new output directory.')
else:
    temporary = metadata_path.with_suffix(f'.tmp{os.getpid()}')
    temporary.write_text(json.dumps(parameters, indent=2))
    os.replace(temporary, metadata_path)

# =============================================================================
# Sampling
# =============================================================================

t0 = last_print = time.perf_counter()

for h, meth in enumerate(methods):
    for tol in tols:
        configs = gen_configs(
            taus, [T], tol=tol, max_iter=max_iter, n_pts=n_pts,
            integrator=integrators[h], gen_gauss=False, AA_beta=1, AA_m=m,
        )[0]                                # one T row: configs[j] is tau index

        for j in range(len(taus)):
            config = configs[j]
            folder = base / meth / f'tau_{taus[j]}' / f'T_{T}' / f'tol_{tol}' / f'len_{length}'
            folder.mkdir(parents=True, exist_ok=True)
            if all((folder / f'run_{i}.npz').exists() for i in range(n_runs)):
                continue

            kernel = gen_hmc_kernel(H, config) if meth == 'LF' else gen_chmc_kernel(H, config)
            scan_sampler = jax.jit(lambda init, keys: jax.lax.scan(kernel, init, keys))

            for i in range(n_runs):
                path = folder / f'run_{i}.npz'
                if path.exists():
                    continue
                key = jr.PRNGKey(i)
                keya, keyb = jr.split(key)
                chain_keys = jr.split(keya, length)
                qp0 = QP(jr.normal(keyb, shape=(2*dim,), dtype=jnp.float64))
                init = [qp0, jnp.float64(0), jnp.asarray(False)]
                jax.block_until_ready((init, chain_keys))

                if i == 0:                      # compile once per configuration, outside timing
                    scan_sampler = scan_sampler.lower(init, chain_keys).compile()

                start = time.perf_counter()
                result = scan_sampler(init, chain_keys)
                jax.block_until_ready(result)
                elapsed = time.perf_counter() - start
                _, (qps, deltaHs, accepted) = result

                temporary = path.with_suffix('.tmp.npz')
                np.savez(
                    temporary, q=np.asarray(qps.q), deltaHs=np.asarray(deltaHs),
                    accepted=np.asarray(accepted), runtime=elapsed,
                    tau=config.τ, T=config.T, N=config.N,
                    tol=tol,
                )
                os.replace(temporary, path)

                # Progress: every 10 s, then every 60 s once the sweep passes 5 minutes.
                now = time.perf_counter()
                if now - last_print >= (10 if now - t0 < 300 else 60):
                    last_print = now
                    print(f'[{now - t0:.0f}s] {meth}: tau={config.τ}, tol={tol}, run={i}, '
                          f'last chain {elapsed:.3g}s', flush=True)

print(f'Done in {time.perf_counter() - t0:.0f}s. Data in {base}')
