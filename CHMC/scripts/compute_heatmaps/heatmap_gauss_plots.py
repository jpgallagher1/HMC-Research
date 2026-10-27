# October 20, 2026
# Compute-time heatmaps for the Gaussian [[1, rho], [rho, 1]] covariance target
# (data from heatmap_gauss.py), including wall time relative to HMC-LF.
# Importable from heatmap.ipynb (functions) or run as a script (saves PNGs).

from pathlib import Path
import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

labels = {'LF': 'HMC-LF', 'AA': 'CHMC-AA', 'Newt': 'CHMC-Newton', 'Newton': 'CHMC-Newton', 'FPI': 'CHMC-FPI'}
fail_rate = 0.01        # a cell is marked [x] if the mean acceptance rate is below this


def load_cells(base, method, T=3.0, length=10_000):
    """Return {(tau, tol): (runtimes, acceptance_rates)} for one method, as in plot_compute_heatmap."""
    root = Path(base) / method
    groups = defaultdict(lambda: ([], []))
    for path in sorted(root.glob("tau_*/T_*/tol_*/len_*/run_*.npz")):
        tau_dir, time_dir, tol_dir, length_dir, _ = path.relative_to(root).parts
        if (not np.isclose(float(time_dir.removeprefix("T_")), T, rtol=1e-12, atol=0)
                or int(length_dir.removeprefix("len_")) != length):
            continue
        with np.load(path, allow_pickle=False) as data:
            runtime = float(data["runtime"].item())
            rate = float(np.mean(data["accepted"]))
        if not np.isfinite(runtime) or runtime < 0:
            raise ValueError(f"Invalid runtime in {path}: {runtime}")
        key = (float(tau_dir.removeprefix("tau_")), float(tol_dir.removeprefix("tol_")))
        groups[key][0].append(runtime)
        groups[key][1].append(rate)
    if not groups:
        raise FileNotFoundError(f"No run files found under {root} for T={T}, length={length}.")
    return groups


def cell_arrays(groups):
    """Arrays with tau as rows and tolerance as columns, both ascending (as in plot_compute_heatmap)."""
    taus = np.array(sorted({tau for tau, _ in groups}))
    tols = np.array(sorted({tol for _, tol in groups}))
    seconds = np.full((len(taus), len(tols)), np.nan)
    accept = np.full(seconds.shape, np.nan)
    for i, tau in enumerate(taus):
        for j, tol in enumerate(tols):
            if (tau, tol) in groups:
                seconds[i, j] = np.mean(groups[tau, tol][0])
                accept[i, j] = np.mean(groups[tau, tol][1])
    return taus, tols, seconds, accept


def draw_heatmap(ax, values, accept, taus, tols, fmt, norm=None, fs=None, cbar_label=None, title=None,
                 seconds=None):
    """seconds: optional mean wall time per cell, printed instead of the value (colors still follow values)."""
    fs = fs or {}
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    image = ax.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto",
                      interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(tols)), labels=[f"{tol:g}" for tol in tols])
    ax.set_yticks(np.arange(len(taus)), labels=[f"{tau:g}" for tau in taus])
    ax.set_xlabel(r"Solver tolerance $\delta$", **fs)
    ax.set_ylabel(r"Step size $\tau$", **fs)
    if title:
        ax.set_title(title, **fs)
    cell_fs = {'fontsize': fs['fontsize'] - 2} if 'fontsize' in fs else {}   # cell text a bit smaller than labels
    for (i, j), value in np.ndenumerate(values):
        if accept[i, j] < fail_rate:       # chain never moved: its wall time is not meaningful
            ax.text(j, i, "[x]", color="red", ha="center", va="center", fontweight="bold", **cell_fs)
        elif np.isfinite(value):
            color = "black" if image.norm(value) > 0.5 else "white"
            text = fmt(value) if seconds is None else f"{seconds[i, j]:.2f} s"
            ax.text(j, i, text, ha="center", va="center", color=color, **cell_fs)
    return image


def plot_mean_runtime_heatmap(base, method, T=3.0, length=10_000, ppt=False):
    """Mean wall time per chain (s) over tau and tolerance, as in heatmap.ipynb."""
    fs = {'fontsize': 16} if ppt else {}
    taus, tols, seconds, accept = cell_arrays(load_cells(base, method, T, length))
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    image = draw_heatmap(ax, seconds, accept, taus, tols, lambda v: f"{v:.2f}", fs=fs,
                         title=f"{labels.get(method, method)}: mean runtime, {length:,} MCMC iterations, T={T:g}")
    fig.colorbar(image, ax=ax).set_label("Mean wall-clock time per chain (s)", **fs)
    return fig, ax, seconds, accept, taus, tols


def plot_runtime_ratio_heatmap(base, methods, baseline='LF', T=3.0, length=10_000, ppt=False,
                               divider=(0.125, 0.25), baseline_tau=0.0625):
    """Wall time of each method divided by HMC's at tau = baseline_tau (above 1 means slower).
    Colors show the ratio; each cell prints the mean wall time (s). Sized to fill 0.9 x 0.7 of a 16:9 slide.

    One reference time is used for every cell. HMC has no solver tolerance, so its panel is a
    single column: the mean over all its saved runs at each tau (every tolerance folder is a
    repeat). Include `baseline` in `methods` to draw that column, e.g. ['AA', 'LF', 'Newt'].
    Cells where the method's chain never moved show [x].
    A red dotted line is drawn between the two tau values in `divider` (None for no line);
    the default marks the leapfrog stability limit 2 sqrt(1 - rho) ~ 0.18 for kappa = 256.
    """
    fs = {'fontsize': 16} if ppt else {}
    base_taus, _, base_seconds, _ = cell_arrays(load_cells(base, baseline, T, length))
    lf_time = np.nanmean(base_seconds[int(np.argmin(np.abs(base_taus - baseline_tau)))])   # one number
    panels = []
    for method in methods:
        taus, tols, seconds, accept = cell_arrays(load_cells(base, method, T, length))
        if method == baseline:          # no solver tolerance: collapse the repeats to one column
            seconds = np.nanmean(seconds, axis=1, keepdims=True)
            accept = np.nanmean(accept, axis=1, keepdims=True)
        panels.append((method, seconds / lf_time, accept, seconds))
    norm = Normalize(vmin=0, vmax=10)          # shared linear scale; ratios above 10 take the top color
    widths = [ratio.shape[1] for _, ratio, _, _ in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(12.0, 5.25),        # 0.9 x 0.7 of a 13.33 x 7.5 in (16:9) slide
                             layout="constrained", squeeze=False, gridspec_kw={'width_ratios': widths})
    for ax, (method, ratio, accept, seconds) in zip(axes[0], panels):
        image = draw_heatmap(ax, ratio, accept, taus, tols[:ratio.shape[1]], lambda v: f"{v:.2g}×",
                             norm=norm, fs=fs, title=labels.get(method, method), seconds=seconds)
        if ppt:
            ax.tick_params(labelsize=14)      # axis tick numbers
        if ax is not axes[0, 0]:            # tau labels on the left panel only
            ax.set_ylabel("")
        if method == baseline:
            ax.set_xticks([])
            ax.set_xlabel("")
        if divider is not None:
            lo, hi = (int(np.argmin(np.abs(taus - tau))) for tau in divider)
            ax.axhline((lo + hi) / 2, color="red", linestyle=":", linewidth=2.5)
    cbar = fig.colorbar(image, ax=axes[0].tolist(), extend="max", ticks=np.arange(0, 11, 2))
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in np.arange(0, 11, 2)])
    if ppt:
        cbar.ax.tick_params(labelsize=14)
    cbar.set_label(rf"Time / {labels.get(baseline, baseline)} at $\tau = {baseline_tau:g}$", **fs)
    if not ppt:                         # on a slide the slide title replaces this
        fig.suptitle(f"Wall time vs HMC, {length:,} MCMC iterations, T={T:g}", **fs)
    return fig, axes, panels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(
        '/Users/johngallagher/Documents/UCMerced/Research/GitHub/HMC-Online/results/heatmap_k256_covmat'))
    parser.add_argument('--ppt', type=lambda x: x.lower() in ('true', '1', 'yes'), default=False)  # True: 16 pt text
    args = parser.parse_args()
    plt.switch_backend('Agg')
    meta = json.loads((args.output / 'metadata.json').read_text())
    T, length = meta['T'], meta['length']
    plotpath = args.output / ('plot_ppt' if args.ppt else 'plot')
    plotpath.mkdir(exist_ok=True)

    others = [m for m in ['AA', 'LF', 'Newt'] if m in meta['methods']]   # FPI left out of the ratio figure
    for method in meta['methods']:
        fig, *_ = plot_mean_runtime_heatmap(args.output, method, T, length, args.ppt)
        fig.savefig(plotpath / f'runtime_{method}.png', dpi=200)
        plt.close(fig)
    fig, *_ = plot_runtime_ratio_heatmap(args.output, others, 'LF', T, length, args.ppt)
    fig.savefig(plotpath / 'runtime_ratio_vs_LF.png', dpi=200)
    plt.close(fig)
    print(f'Plots saved to {plotpath}')
