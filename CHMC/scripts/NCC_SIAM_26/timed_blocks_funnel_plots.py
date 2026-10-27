# October 20, 2026
# Sliced Wasserstein error plots for Neal's funnel, in the style of timed_blocks_plots.py.
# Plotting only: reads OUTPUT/sw_data written by timed_blocks_funnel_sw.py and computes nothing.

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

base = Path('/Users/johngallagher/Documents/UCMerced/Research/GitHub/HMC-Online/results/neals_funnel_blocks')
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=base)
parser.add_argument('--ppt', type=lambda x: x.lower() in ('true', '1', 'yes'), default=False)   # True: 16 pt text
args = parser.parse_args()
base = args.output
sw_base = base / 'sw_data'
plotpath = base / 'nanmeans' / ('plot_ppt' if args.ppt else 'plot')   # NaN-ignoring means, kept apart from the older plots
plotpath.mkdir(parents=True, exist_ok=True)

with open(base / 'metadata.json') as f:
    metadata = json.load(f)
with open(sw_base / 'settings.json') as f:
    sw_settings = json.load(f)

methods = metadata['methods']
taus = metadata['taus']
Ts = metadata['Ts']
dim = metadata['dim']
sw_p = sw_settings['p']
markers = {'LF': '-o', 'AA': '-*', 'Newt': '-^', 'Newton': '-^', 'N_K': '-^', 'FPI': '-v'}
labels = {'LF': 'HMC-LF', 'AA': 'CHMC-AA', 'Newt': r'CHMC-Newton', 'Newton': r'CHMC-Newton', 'N_K': r'CHMC-Newton-Krylov', 'FPI': 'CHMC-FPI'}
colors = {'AA': 'C0', 'LF': 'C1', 'FPI': 'C2', 'Newt': 'C3', 'Newton': 'C3', 'N_K': 'C4'}   # as in the p-chi DRAFT plot
plot_order = [meth for meth in ['AA', 'LF', 'FPI', 'Newt', 'Newton', 'N_K'] if meth in methods]
legend_order = [meth for meth in ['LF', 'FPI', 'AA', 'Newt', 'Newton', 'N_K'] if meth in methods]

# =============================================================================
# Load the saved sliced Wasserstein values
# =============================================================================

def sw_path(meth, j, k, i):
    return sw_base / meth / f'tau_{taus[j]}' / f'T_{Ts[k]}' / f'run_{i}.npz'


files = list(sw_base.glob('*/tau_*/T_*/run_*.npz'))
if not files:
    raise ValueError(f'No sliced Wasserstein data found in {sw_base}; run timed_blocks_funnel_sw.py first.')
n_runs = max(int(path.stem.split('_')[1]) for path in files) + 1
lens = np.unique(np.concatenate([np.load(path)['lens'] for path in files]))

# Same axis order as the p-chi plotting notebook: tau, T, length, run.
shape = (len(taus), len(Ts), len(lens), n_runs)
hmc_sw_metrics = {meth: np.full(shape, np.nan) for meth in methods}
hmc_runtimes = {meth: np.full(shape, np.nan) for meth in methods}
for meth in methods:
    for j in range(len(taus)):
        for k in range(len(Ts)):
            for i in range(n_runs):
                path = sw_path(meth, j, k, i)
                if not path.exists():
                    continue
                with np.load(path) as data:
                    at = np.searchsorted(lens, data['lens'])
                    hmc_sw_metrics[meth][j, k, at, i] = data['sw']
                    hmc_runtimes[meth][j, k, at, i] = data['runtime']

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
        for xaxis, scale in [('iter', 'loglog'), ('time', 'semilog'), ('time', 'loglog')]:
            fig = plt.figure(figsize=(8, 6))
            handles = {}
            for meth in plot_order:
                errors = hmc_sw_metrics[meth][j, k, :, :]
                if not np.isfinite(errors).any():
                    continue
                times = hmc_runtimes[meth][j, k, :, :]
                x = np.broadcast_to(lens[:, None], errors.shape) if xaxis == 'iter' else times
                plot = plt.loglog if scale == 'loglog' else plt.semilogy
                for i in range(errors.shape[-1]):           # each run, NaNs dropped so the line connects
                    ok = np.isfinite(errors[:, i]) & np.isfinite(x[:, i])
                    plot(x[ok, i], errors[ok, i], markers[meth], color=colors[meth], alpha=0.15)

                # Mean over the runs that have data at each length (NaNs ignored), so the mean
                # continues as far as the longest extended run.
                valid = np.isfinite(errors) & np.isfinite(x)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)     # lengths with no runs give NaN
                    avg_error = np.nanmean(np.where(valid, errors, np.nan), axis=-1)
                    avg_x = np.nanmean(np.where(valid, x, np.nan), axis=-1)
                ok = np.isfinite(avg_error) & np.isfinite(avg_x)
                handles[meth], = plot(avg_x[ok], avg_error[ok], markers[meth], color=colors[meth],
                                      label=f'Mean {labels[meth]}', alpha=1)

            title = rf"Sliced $W_{sw_p}$ Error, Neal's funnel ($d = {dim}$)"
            subtitle = '\n' + rf'$\tau = {taus[j]}$, $T = {Ts[k]}$'
            plt.title(title + subtitle, **fs)
            plt.xlabel('MCMC Iterations' if xaxis == 'iter' else 'Time (s)', **fs)
            plt.ylabel(ylabel, **fs)
            plt.grid(which='minor', alpha=0.3)
            plt.grid(which='major', alpha=0.3)
            order = [meth for meth in legend_order if meth in handles]
            plt.legend([handles[meth] for meth in order], [f'Mean {labels[meth]}' for meth in order], **fs)
            if args.ppt:
                plt.tick_params(labelsize=14)      # axis tick numbers
            plt.tight_layout()
            folder = plotpath / 'sw' / f'tau_{taus[j]}'.replace('.', '_') / f'T_{Ts[k]}'.replace('.', '_')
            folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(folder / f'funnel_sw{sw_p}_{xaxis}_{scale}.png', dpi=200)
            plt.close(fig)

print(f'Plots saved to {plotpath}')
