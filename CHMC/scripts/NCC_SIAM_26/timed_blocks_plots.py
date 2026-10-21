# October 18, 2026
# Covariance and Wasserstein-1 error plots in the style of W1plots-p=4_d=400_m=2_centered.ipynb.

from pathlib import Path
import argparse
import json

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

base = Path('/Users/johngallagher/Documents/UCMerced/Research/GitHub/HMC-Online/results/cov_k256_blocks_covmat')
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=base)
parser.add_argument('--which', choices=['cov', 'w1'], default='cov')   # which metric to plot
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
kappa = metadata['kappa']
rho = metadata['rho']      # (kappa - 1)/(kappa + 1)
block_size = metadata['block_size']
tol = metadata['tol']              # solver tolerance
max_iter = metadata['max_iter']    # solver iteration cap (Newton K; AA and FPI cap)
true_cov = np.asarray(metadata['covariance'])
markers = {'LF': '-o', 'AA': '-*', 'Newt': '-^', 'Newton': '-^', 'FPI': '-v'}
labels = {'LF': 'HMC-LF', 'AA': 'CHMC-AA', 'Newt': r'CHMC-Newt', 'Newton': r'CHMC-Newt', 'FPI': 'CHMC-FPI'}

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
# All are multiples of block_size, so each is a saved block boundary.
# Extended chains (matched to Newt's wall time) are also evaluated every 5000 past 10_000.
lens = np.array([n for n in [*range(200, 1000, 200), *range(1000, 10_001, 1000),
                             *range(15_000, nblocks * block_size + 1, 5000)]
                 if n <= nblocks * block_size])

# W1 is computed along the two principal axes of the target (the eigenvectors of
# the covariance), where the chain is hardest/easiest to resolve for large kappa;
# the reported error is the larger of the two 1D W1 distances.
axes_dirs, axis_var = np.linalg.eigh(true_cov)[1], np.linalg.eigvalsh(true_cov)
axis_std = np.sqrt(axis_var)                                             # std along each axis


def w1_to_gaussian(x, std):
    """Exact W1 between the empirical measure of x and N(0, std^2): integral of |F_n - F|."""
    x = np.sort(x)
    n = len(x)
    z = x / std
    G = std * (z * norm.cdf(z) + norm.pdf(z))      # antiderivative of the Gaussian cdf
    a, b = x[:-1], x[1:]
    c = np.arange(1, n) / n                        # F_n on [x_(i), x_(i+1)]
    Ga, Gb = G[:-1], G[1:]
    xs = np.clip(std * norm.ppf(c), a, b)          # where F crosses c inside the interval
    Gs = std * (xs / std * norm.cdf(xs / std) + norm.pdf(xs / std))
    # integral of |c - F| over [a, b], split at xs: F < c on [a, xs], F > c on [xs, b]
    inner = (c * (xs - a) - (Gs - Ga)) + ((Gb - Gs) - c * (b - xs))
    left = G[0]                                    # integral of F on (-inf, x_(1)]
    right = std * (norm.pdf(z[-1]) - z[-1] * norm.cdf(-z[-1]))   # integral of 1-F on [x_(n), inf)
    return left + inner.sum() + right


# Same axis order as the p-chi plotting notebook: tau, T, length, run.
shape = (len(taus), len(Ts), len(lens), n_runs)
hmc_cov_metrics = {meth: np.full(shape, np.nan) for meth in methods}
hmc_full_cov_metrics = {meth: np.full(shape, np.nan) for meth in methods}
hmc_w1_metrics = {meth: np.full(shape, np.nan) for meth in methods}
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
                if not np.isfinite(full_chain).all():
                    raise ValueError(f'Nonfinite positions in {folder}')
                # Cumulative runtime at each saved block edge; linear within a block.
                edges = np.arange(len(blocks) + 1) * block_size
                cum_time = np.concatenate([[0.0], np.cumsum(block_times)])

                for l, length in enumerate(lens):
                    if length > len(full_chain):
                        break
                    chain = full_chain[:length]
                    hmc_runtimes[meth][j, k, l, i] = np.interp(length, edges, cum_time)
                    cov_error = np.cov(chain, rowvar=False, ddof=1) - true_cov
                    hmc_cov_metrics[meth][j, k, l, i] = np.max(np.abs(np.diag(cov_error)))
                    hmc_full_cov_metrics[meth][j, k, l, i] = np.max(np.abs(cov_error))
                    if args.which == 'w1':
                        chain_proj = chain @ axes_dirs
                        hmc_w1_metrics[meth][j, k, l, i] = max(
                            w1_to_gaussian(chain_proj[:, c], axis_std[c]) for c in range(chain.shape[1]))

# One file per metric, so running --which cov never overwrites saved W1 values.
saved = ({'cov_metric': hmc_cov_metrics, 'full_cov_metric': hmc_full_cov_metrics}
         if args.which == 'cov' else {'w1_metric': hmc_w1_metrics})
np.savez(
    base / f'{args.which}_metrics.npz', lens=lens, taus=taus, Ts=Ts,
    **{f'{meth}_{name}': data[meth] for name, data in saved.items() for meth in methods},
    **{f'{meth}_runtime': hmc_runtimes[meth] for meth in methods},
)

# =============================================================================
# Plot individual runs and means, as in the p-chi plots
# =============================================================================

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['mathtext.fontset'] = 'cm'

fs = {'fontsize': 16} if args.ppt else {}     # title, axis labels and legend size for slides

plot_metrics = {
    'cov': [('diag', hmc_cov_metrics, 'Maximum Covariance Diagonal Error', 'Covariance Error'),
            ('full', hmc_full_cov_metrics, 'Maximum Covariance Entry Error', 'Covariance Error')],
    'w1': [('w1', hmc_w1_metrics, r'Wasserstein $W_1$ Error', r'$W_1$ Error')],
}[args.which]

for metric_name, metric_data, ylabel, metric_title in plot_metrics:
    for j in range(len(taus)):
        for k in range(len(Ts)):
            if not any(np.isfinite(metric_data[meth][j, k]).any() for meth in methods):
                continue
            # loglog against iterations, semilog against time
            for xaxis, scale in [('iter', 'loglog'), ('time', 'semilog')]:
                fig = plt.figure(figsize=(8, 6))
                for Cidx, meth in enumerate(methods):
                    errors = metric_data[meth][j, k, :, :]
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

                sym = r'\alpha' if args.ppt else r'\rho'     # slides call the matrix perturbation alpha
                title = rf'{metric_title}, covariance ${sym} = {rho:.4f}$'
                subtitle = '\n' + rf'$\tau = {taus[j]}$, $T = {Ts[k]}$'
                if args.ppt:        # slides also state the solver settings
                    subtitle += '\n' + rf'solver tolerance $\delta = {tol:g}$, max iterations: ${max_iter}$'
                plt.title(title + subtitle, **fs)
                plt.xlabel('MCMC Iterations' if xaxis == 'iter' else 'Time (s)', **fs)
                plt.ylabel(ylabel, **fs)
                plt.grid(which='minor', alpha=0.3)
                plt.grid(which='major', alpha=0.3)
                plt.legend(**fs)
                if args.ppt:
                    plt.tick_params(labelsize=14)      # axis tick numbers
                plt.tight_layout()
                folder = plotpath / args.which / f'tau_{taus[j]}'.replace('.', '_') / f'T_{Ts[k]}'.replace('.', '_')
                folder.mkdir(parents=True, exist_ok=True)
                plt.savefig(folder / f'k{kappa}_{metric_name}_{xaxis}_{scale}.png', dpi=200)
                plt.close(fig)

print(f'Plots saved to {plotpath}')
