# October 20, 2026
# Sliced Wasserstein error plots for Neal's funnel (timed_blocks_funnel.py output),
# in the style of timed_blocks_plots.py.

from pathlib import Path
import argparse
import json

import numpy as np
import ot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

n_ref = 10_000              # exact funnel samples used as the reference measure
n_projections = 100         # random directions; the same ones are used for every chain
sw_p = 1                    # sliced W_p: 1 gives sliced W_1 (mean over directions of 1D W_1)
ref_seed = 0

base = Path('/Users/johngallagher/Documents/UCMerced/Research/GitHub/HMC-Online/results/neals_funnel_blocks')
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=base)
parser.add_argument('--ppt', type=lambda x: x.lower() in ('true', '1', 'yes'), default=False)   # True: 16 pt text
args = parser.parse_args()
base = args.output
plotpath = base / ('plot_ppt' if args.ppt else 'plot')
plotpath.mkdir(exist_ok=True)

with open(base / 'metadata.json') as f:
    metadata = json.load(f)

methods = metadata['methods']
taus = metadata['taus']
Ts = metadata['Ts']
dim = metadata['dim']
block_size = metadata['block_size']
markers = {'LF': '-o', 'AA': '-*', 'Newt': '-^', 'Newton': '-^', 'N_K': '-^', 'FPI': '-v'}
labels = {'LF': 'HMC-LF', 'AA': 'CHMC-AA', 'Newt': r'CHMC-Newt', 'Newton': r'CHMC-Newt', 'N_K': r'CHMC-Newt', 'FPI': 'CHMC-FPI'}

# =============================================================================
# Load the saved blocks
# =============================================================================

folders = list(base.glob('*/tau_*/T_*/run_*'))
if not folders:
    raise ValueError('No saved runs found.')
n_runs = max(int(folder.name.split('_')[1]) for folder in folders) + 1
nblocks = max(len(list(folder.glob('block_*.npz'))) for folder in folders)
if nblocks == 0:
    raise ValueError('No saved blocks found.')
# Chain lengths to evaluate: 200, 400, 600, 800, 1000, 2000, ..., 10_000 (up to what was saved).
# Extended chains (matched to Newt's wall time) are also evaluated every 5000 past 10_000.
lens = np.array([n for n in [*range(200, 1000, 200), *range(1000, 10_001, 1000),
                             *range(15_000, nblocks * block_size + 1, 5000)]
                 if n <= nblocks * block_size])

# Exact funnel samples: y ~ N(0, 3^2), x_i | y ~ N(0, exp(y)) for the other dim - 1 coordinates.
rng = np.random.default_rng(ref_seed)
y = rng.normal(0.0, 3.0, size=n_ref)
x = rng.normal(size=(n_ref, dim - 1)) * np.exp(y / 2)[:, None]
xt = np.column_stack([y, x])                                             # (n_ref, dim)

# Same axis order as the p-chi plotting notebook: tau, T, length, run.
shape = (len(taus), len(Ts), len(lens), n_runs)
hmc_sw_metrics = {meth: np.full(shape, np.nan) for meth in methods}
hmc_runtimes = {meth: np.full(shape, np.nan) for meth in methods}

for meth in methods:
    for j in range(len(taus)):
        for k in range(len(Ts)):
            for i in range(n_runs):
                folder = base / meth / f'tau_{taus[j]}' / f'T_{Ts[k]}' / f'run_{i}'
                blocks, block_times = [], []
                for l in range(nblocks):
                    path = folder / f'block_{l:08d}.npz'
                    if not path.exists():
                        break
                    with np.load(path) as result:
                        # Retain repeated positions after rejection in the chain.
                        blocks.append(result['q'])
                        block_times.append(float(result['runtime']))
                if not blocks:
                    continue
                full_chain = np.concatenate(blocks, axis=0)
                # Cumulative runtime at each saved block edge; linear within a block.
                edges = np.arange(len(blocks) + 1) * block_size
                cum_time = np.concatenate([[0.0], np.cumsum(block_times)])

                for l, length in enumerate(lens):
                    if length > len(full_chain):
                        break
                    chain = full_chain[:length]
                    hmc_runtimes[meth][j, k, l, i] = np.interp(length, edges, cum_time)
                    if not np.isfinite(chain).all():
                        print(f'Nonfinite positions in {folder} by length {length}; metric left NaN')
                        continue
                    hmc_sw_metrics[meth][j, k, l, i] = ot.sliced_wasserstein_distance(
                        chain, xt, n_projections=n_projections, p=sw_p, seed=ref_seed)

np.savez(
    base / 'sw_metrics.npz', lens=lens, taus=taus, Ts=Ts,
    **{f'{meth}_sw_metric': hmc_sw_metrics[meth] for meth in methods},
    **{f'{meth}_runtime': hmc_runtimes[meth] for meth in methods},
)

# =============================================================================
# Plot individual runs and means, as in the p-chi plots
# =============================================================================

fs = {'fontsize': 16} if args.ppt else {}     # title, axis labels and legend size for slides
ylabel = rf'Sliced Wasserstein $W_{sw_p}$ Error'

for j in range(len(taus)):
    for k in range(len(Ts)):
        if not any(np.isfinite(hmc_sw_metrics[meth][j, k]).any() for meth in methods):
            continue
        # loglog against iterations, semilog against time
        for xaxis, scale in [('iter', 'loglog'), ('time', 'semilog')]:
            fig = plt.figure(figsize=(8, 6))
            for Cidx, meth in enumerate(methods):
                errors = hmc_sw_metrics[meth][j, k, :, :]
                if not np.isfinite(errors).any():
                    continue
                times = hmc_runtimes[meth][j, k, :, :]
                x = np.broadcast_to(lens[:, None], errors.shape) if xaxis == 'iter' else times
                plot = plt.loglog if scale == 'loglog' else plt.semilogy
                plot(x, errors, markers[meth], color=f'C{Cidx}', alpha=0.15)

                valid = np.isfinite(errors) & np.isfinite(x)
                counts = valid.sum(axis=-1)
                # Average only where every run has data; extended runs end at different
                # lengths, and a mean over a shrinking subset would jump.
                counts = np.where(counts == errors.shape[-1], counts, 0)
                avg_error = np.divide(np.where(valid, errors, 0).sum(axis=-1), counts,
                                      out=np.full(len(lens), np.nan), where=counts > 0)
                avg_x = np.divide(np.where(valid, x, 0).sum(axis=-1), counts,
                                  out=np.full(len(lens), np.nan), where=counts > 0)
                plot(avg_x, avg_error, markers[meth], color=f'C{Cidx}',
                     label=f'Mean {labels[meth]}', alpha=1)

            title = rf"Sliced $W_{sw_p}$ Error, Neal's funnel ($d = {dim}$)"
            subtitle = '\n' + rf'$\tau = {taus[j]}$, $T = {Ts[k]}$'
            plt.title(title + subtitle, **fs)
            plt.xlabel('MCMC Iterations' if xaxis == 'iter' else 'Time (s)', **fs)
            plt.ylabel(ylabel, **fs)
            plt.grid(which='minor', alpha=0.3)
            plt.grid(which='major', alpha=0.3)
            plt.legend(**fs)
            if args.ppt:
                plt.tick_params(labelsize=14)      # axis tick numbers
            plt.tight_layout()
            folder = plotpath / 'sw' / f'tau_{taus[j]}'.replace('.', '_') / f'T_{Ts[k]}'.replace('.', '_')
            folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(folder / f'funnel_sw{sw_p}_{xaxis}_{scale}.png', dpi=200)
            plt.close(fig)

print(f'Plots saved to {plotpath}')
