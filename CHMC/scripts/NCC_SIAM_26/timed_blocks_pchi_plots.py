# October 26, 2026
# W1 error plots for p-chi (p, d read from metadata) in the style of timed_blocks_plots.py.
# Reads OUTPUT written by timed_blocks_pchi.py. W1 as in W1plots-p=4_d=400_m=2_centered.ipynb:
# ot.wasserstein_1d(|q| of accepted states, 10_000 exact p-chi draws from PRNGKey(-1)).

from pathlib import Path
import argparse
import json

import jax
jax.config.update('jax_enable_x64', True)
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import ot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

results = Path('/Users/johngallagher/Documents/UCMerced/Research/GitHub/HMC-Online/results')
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=int, default=4)
parser.add_argument('--d', type=int, default=400)
parser.add_argument('--output', type=Path, default=None)          # default: results/pchi_p{p}_d{d}_blocks
parser.add_argument('--ppt', type=lambda x: x.lower() in ('true', '1', 'yes'), default=False)   # True: 16 pt text
args = parser.parse_args()
base = args.output or results / f'pchi_p{args.p}_d{args.d}_blocks'
plotpath = base / ('plot_ppt' if args.ppt else 'plot')
plotpath.mkdir(exist_ok=True)

with open(base / 'metadata.json') as f:
    metadata = json.load(f)

methods = metadata['methods']
taus = metadata['taus']
Ts = metadata['Ts']
p = metadata['p']
d = metadata['d']
block_size = metadata['block_size']
tol = metadata['tol']
max_iter = metadata['max_iter']
AA_m = metadata['AA_m']
n_pts = metadata['n_pts']
plot_order = [meth for meth in ['AA', 'LF', 'FPI', 'Newton'] if meth in methods]   # colors as in the DRAFT plot
legend_order = [meth for meth in ['LF', 'FPI', 'AA', 'Newton'] if meth in methods]
markers = {'LF': '-o', 'AA': '-*', 'Newt': '-^', 'Newton': '-^', 'FPI': '-v'}
labels = {'LF': 'HMC-LF', 'AA': 'CHMC-AA', 'Newt': r'CHMC-Newton', 'Newton': r'CHMC-Newton', 'FPI': 'CHMC-FPI'}

# =============================================================================
# Reference samples, same as the notebook
# =============================================================================

n_ref = 10_000
def sample_pchi(key, n, d, p, dtype=jnp.float64):
    y = jr.gamma(key, d/p, shape=(n,), dtype=dtype)
    return (p*y)**(1/p)
xt = np.asarray(sample_pchi(jr.PRNGKey(-1), n_ref, d, p))     # (n_ref,)

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
# Notebook lengths 100, 178, 316, ..., 10_000 (4 per decade), continued at the same spacing
# for extended chains up to what was saved.
n_max = nblocks * block_size
lens = np.unique(np.round(10**np.arange(2, np.log10(n_max) + 1e-9, 0.25)).astype(int))
lens = lens[lens <= n_max]

# Same axis order as the p-chi plotting notebook: tau, T, length, run.
shape = (len(taus), len(Ts), len(lens), n_runs)
hmc_w1_metrics = {meth: np.full(shape, np.nan) for meth in methods}           # accepted states only, as the notebook
hmc_w1_retained_metrics = {meth: np.full(shape, np.nan) for meth in methods}  # every retained state, as timed_blocks_plots.py
hmc_runtimes = {meth: np.full(shape, np.nan) for meth in methods}

for meth in methods:
    for j in range(len(taus)):
        for k in range(len(Ts)):
            for i in range(n_runs):
                folder = base / meth / f'tau_{taus[j]}' / f'T_{Ts[k]}' / f'run_{i}'
                qs, accs, block_times = [], [], []
                for l in range(nblocks):
                    path = folder / f'block_{l:08d}.npz'
                    if not path.exists():
                        break
                    with np.load(path) as result:
                        qs.append(np.abs(result['q'][:, 0]))      # |q|, radius
                        accs.append(result['accepted'])
                        block_times.append(float(result['runtime']))
                if not qs:
                    continue
                full_q, full_acc = np.concatenate(qs), np.concatenate(accs)
                if not np.isfinite(full_q).all():
                    raise ValueError(f'Nonfinite positions in {folder}')
                # Cumulative runtime at each saved block edge; linear within a block.
                edges = np.arange(len(qs) + 1) * block_size
                cum_time = np.concatenate([[0.0], np.cumsum(block_times)])

                for l, length in enumerate(lens):
                    if length > len(full_q):
                        break
                    q, acc = full_q[:length], full_acc[:length]
                    hmc_runtimes[meth][j, k, l, i] = np.interp(length, edges, cum_time)
                    hmc_w1_retained_metrics[meth][j, k, l, i] = ot.wasserstein_1d(q, xt)
                    if acc.any():
                        hmc_w1_metrics[meth][j, k, l, i] = ot.wasserstein_1d(q[acc], xt)

np.savez(
    base / 'w1_metrics.npz', lens=lens, taus=taus, Ts=Ts,
    **{f'{meth}_w1_metric': hmc_w1_metrics[meth] for meth in methods},
    **{f'{meth}_w1_retained_metric': hmc_w1_retained_metrics[meth] for meth in methods},
    **{f'{meth}_runtime': hmc_runtimes[meth] for meth in methods},
)

# =============================================================================
# Plot individual runs and means, as in the p-chi plots
# =============================================================================

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['mathtext.fontset'] = 'cm'

fs = {'fontsize': 16} if args.ppt else {}     # title, axis labels and legend size for slides

plot_metrics = [('w1', hmc_w1_metrics, 'accepted states'),
                ('w1_retained', hmc_w1_retained_metrics, 'all retained states')]

for metric_name, metric_data, states in plot_metrics:
    for j in range(len(taus)):
        for k in range(len(Ts)):
            if not any(np.isfinite(metric_data[meth][j, k]).any() for meth in methods):
                continue
            # loglog against iterations, semilog and loglog against time
            for xaxis, scale in [('iter', 'loglog'), ('time', 'semilog'), ('time', 'loglog')]:
                fig = plt.figure(figsize=(8, 6))
                handles = {}
                for meth in plot_order:
                    Cidx = ['AA', 'LF', 'FPI', 'Newton'].index(meth)
                    errors = metric_data[meth][j, k, :, :]
                    if not np.isfinite(errors).any():
                        continue
                    times = hmc_runtimes[meth][j, k, :, :]
                    times = np.where(times > 0, times, np.nan)  # log x: non-positive runtimes are invalid
                    x = np.broadcast_to(lens[:, None], errors.shape) if xaxis == 'iter' else times
                    plot = plt.loglog if scale == 'loglog' else plt.semilogy
                    for i in range(errors.shape[-1]):           # each run, NaNs dropped so the line connects
                        ok = np.isfinite(errors[:, i]) & np.isfinite(x[:, i])
                        plot(x[ok, i], errors[ok, i], markers[meth], color=f'C{Cidx}', alpha=0.15)

                    valid = np.isfinite(errors) & np.isfinite(x)
                    counts = valid.sum(axis=-1)
                    # Average only where every run has data; extended runs end at different
                    # lengths, and a mean over a shrinking subset would jump.
                    counts = np.where(counts == errors.shape[-1], counts, 0)
                    avg_error = np.divide(np.where(valid, errors, 0).sum(axis=-1), counts,
                                          out=np.full(len(lens), np.nan), where=counts > 0)
                    avg_x = np.divide(np.where(valid, x, 0).sum(axis=-1), counts,
                                      out=np.full(len(lens), np.nan), where=counts > 0)
                    ok = np.isfinite(avg_error) & np.isfinite(avg_x)
                    handles[meth], = plot(avg_x[ok], avg_error[ok], markers[meth], color=f'C{Cidx}',
                                          label=f'Mean {labels[meth]}', alpha=1)

                title = rf'Wasserstein $W_1$ Error, p-chi $p = {p}$, $d = {d}$'
                subtitle = '\n' + rf'$\tau = {taus[j]}$, $T = {Ts[k]}$, {states}'
                subtitle += '\n' + rf'AA window ${AA_m}$, tol $= {tol:g}$, max iter $= {max_iter}$, AVF nodes $= {n_pts}$'
                plt.title(title + subtitle, **fs)
                plt.xlabel('MCMC Iterations' if xaxis == 'iter' else 'Time (s)', **fs)
                plt.ylabel(r'Wasserstein $W_1$ Error', **fs)
                plt.grid(which='minor', alpha=0.3)
                plt.grid(which='major', alpha=0.3)
                order = [meth for meth in legend_order if meth in handles]
                plt.legend([handles[meth] for meth in order], [f'Mean {labels[meth]}' for meth in order], **fs)
                if args.ppt:
                    plt.tick_params(labelsize=14)      # axis tick numbers
                plt.tight_layout()
                folder = plotpath / metric_name / f'tau_{taus[j]}'.replace('.', '_') / f'T_{Ts[k]}'.replace('.', '_')
                folder.mkdir(parents=True, exist_ok=True)
                plt.savefig(folder / f'pchi_p{p}_d{d}_{metric_name}_{xaxis}_{scale}.png', dpi=200)
                plt.close(fig)

print(f'Plots saved to {plotpath}')
