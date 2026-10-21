# October 18, 2026
# Gaussian experiment: covariance [[1, rho], [rho, 1]], kappa = 512.
# Continue each chain in timed blocks of 200 and save each block immediately.

import sys
from pathlib import Path
import argparse
import json
import os
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
jax.config.update("jax_enable_x64", True)
from jax import jit
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
Ts = [1.0, 2.0, 3.0, 4.0, 5.0]

block_size = 200
num_samples = 10_000
match_time_to = 'Newt'         # methods below continue past num_samples until their
extend_methods = ['LF', 'AA', 'FPI']   # cumulative wall time reaches this method's at num_samples
seed = 1
tol = 1e-3
max_iter = 10                  # K cap for Newton; iteration cap for AA and FPI
m = 3
n_pts = 4
base = Path(f'/Users/johngallagher/Documents/UCMerced/Research/GitHub/HMC-Online/results/cov_k{kappa}_blocks_covmat')

# Optional overrides for short runs; experiment settings live above.
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=base)
parser.add_argument('--samples', type=int, default=num_samples)
parser.add_argument('--runs', type=int, default=n_runs)
parser.add_argument('--taus', type=float, nargs='+', default=taus)
parser.add_argument('--Ts', type=float, nargs='+', default=Ts)
args = parser.parse_args()
base = args.output
num_samples = args.samples
n_runs = args.runs
taus = args.taus
Ts = args.Ts

assert block_size > 0 and num_samples > 0 and num_samples % block_size == 0
assert n_runs > 0 and max_iter > 0 and m > 0 and n_pts > 0
assert all(np.isfinite(x) and x > 0 for x in [*taus, *Ts, kappa, tol])

# =============================================================================
# Derived quantities and metadata
# =============================================================================

true_cov = gen_perturb_mat(dim, rho)  # [[1, rho], [rho, 1]] is the COVARIANCE
precision = jnp.linalg.inv(true_cov)   # the Hamiltonian takes the precision
H = gaussian_hamiltonian(precision, mass_inv=Mass_inv)
lens = np.arange(block_size, num_samples + 1, block_size)

print(f'JAX devices: {jax.devices()}')
print(f'dim: {dim}, kappa: {kappa}, rho: {rho}, block_size: {block_size}')

base.mkdir(parents=True, exist_ok=True)
parameters = dict(
    schema=2, dim=dim, kappa=kappa, rho=rho, matrix_role='covariance',
    precision=np.asarray(precision).tolist(), covariance=np.asarray(true_cov).tolist(),
    methods=methods, integrators=integrators, taus=taus, Ts=Ts,
    block_size=block_size, seed=seed, tol=tol, max_iter=max_iter,
    AA_m=m, n_pts=n_pts, jax_version=jax.__version__, backend=jax.default_backend(),
)
metadata_path = base / 'metadata.json'
if metadata_path.exists():
    if json.loads(metadata_path.read_text()) != parameters:
        raise ValueError('Saved settings differ. Use a new output directory.')
else:
    temporary = metadata_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(parameters, indent=2))
    os.replace(temporary, metadata_path)

# =============================================================================
# Sampling
# =============================================================================

t0 = last_print = time.perf_counter()

# The reference method runs first so the others can match its wall time (keys still use h).
for h, meth in sorted(enumerate(methods), key=lambda hm: hm[1] != match_time_to):
    configs = gen_configs(
        taus, Ts, tol=tol, max_iter=max_iter, n_pts=n_pts,
        integrator=integrators[h], gen_gauss=False, AA_beta=1, AA_m=m,
    )                                   # configs[k][j]: T index, tau index

    for j in range(len(taus)):
        for k in range(len(Ts)):
            config = configs[k][j]
            kernel = gen_hmc_kernel(H, config) if meth == 'LF' else gen_chmc_kernel(H, config)
            scan_sampler = jit(lambda init, keys: jax.lax.scan(kernel, init, keys))

            for i in range(n_runs):
                folder = base / meth / f'tau_{taus[j]}' / f'T_{Ts[k]}' / f'run_{i}'
                folder.mkdir(parents=True, exist_ok=True)

                key = jr.PRNGKey(seed)
                for index in [h, j, k, i]:
                    key = jr.fold_in(key, index)
                key, init_key = jr.split(key)
                qp0 = QP(jr.normal(init_key, shape=(2*dim,), dtype=jnp.float64))
                init = [qp0, jnp.float64(0), jnp.asarray(False)]
                elapsed = 0.0
                start_block = 0

                # Restore the last committed block without replacing earlier data.
                files = sorted(folder.glob('block_*.npz'))
                for l, path in enumerate(files):
                    if path.name != f'block_{l:08d}.npz':
                        raise ValueError(f'Missing block in {folder}')
                if files:
                    with np.load(files[-1]) as data:
                        init = [QP(jnp.asarray(data['end_qp'])),
                                jnp.asarray(data['end_deltaH']), jnp.asarray(data['end_accepted'])]
                        key = jnp.asarray(data['next_key'])
                        elapsed = float(data['cumulative_runtime'])
                        start_block = len(files)
                        assert int(data['stop']) == start_block * block_size

                # Extended methods keep going past num_samples until their cumulative wall time
                # matches the reference method's, for the same tau, T and run, at num_samples.
                target_time = None
                if meth in extend_methods:
                    ref_files = sorted((base / match_time_to / f'tau_{taus[j]}' / f'T_{Ts[k]}'
                                        / f'run_{i}').glob('block_*.npz'))
                    if len(ref_files) < len(lens):
                        raise ValueError(f'{match_time_to} must finish {num_samples} samples first: '
                                         f'tau={taus[j]}, T={Ts[k]}, run={i}')
                    with np.load(ref_files[len(lens) - 1]) as data:
                        target_time = float(data['cumulative_runtime'])

                def more_blocks(l, elapsed):
                    return l < len(lens) or (target_time is not None and elapsed < target_time)

                if not more_blocks(start_block, elapsed):
                    continue

                # Warm up / compile without advancing the saved chain or key.
                warm_keys = jr.split(jr.fold_in(key, 0), block_size)
                jax.block_until_ready(scan_sampler(init, warm_keys))

                l = start_block
                while more_blocks(l, elapsed):
                    next_key, block_key = jr.split(key)
                    chain_keys = jr.split(block_key, block_size)
                    jax.block_until_ready(chain_keys)
                    start_qp = np.asarray(init[0].x)

                    # Timed block: continue from the preceding block's final state.
                    start = time.perf_counter()
                    last, (qps, deltaHs, accepted) = scan_sampler(init, chain_keys)
                    jax.block_until_ready((last, qps, deltaHs, accepted))
                    runtime = time.perf_counter() - start
                    elapsed += runtime

                    # Save only the new 200 transitions, plus the state for restart.
                    path = folder / f'block_{l:08d}.npz'
                    temporary = path.with_suffix('.tmp')
                    with temporary.open('wb') as f:
                        np.savez(
                            f, q=np.asarray(qps.q), p=np.asarray(qps.p),
                            deltaHs=np.asarray(deltaHs), accepted=np.asarray(accepted),
                            runtime=runtime, cumulative_runtime=elapsed,
                            start=l*block_size, stop=(l+1)*block_size, start_qp=start_qp,
                            end_qp=np.asarray(last[0].x), end_deltaH=np.asarray(last[1]),
                            end_accepted=np.asarray(last[2]), next_key=np.asarray(next_key),
                            tau=config.τ, T=config.T, N=config.N,
                        )
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temporary, path)

                    init = last
                    key = next_key
                    # Progress: every 10 s, then every 60 s once the sweep passes 5 minutes.
                    now = time.perf_counter()
                    if now - last_print >= (10 if now - t0 < 300 else 60):
                        last_print = now
                        print(f'[{now - t0:.0f}s] {meth}: tau={config.τ}, T={config.T}, run={i}, '
                              f'length={(l + 1) * block_size}', flush=True)
                    l += 1
