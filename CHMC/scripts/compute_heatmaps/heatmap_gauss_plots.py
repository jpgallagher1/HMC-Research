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
from matplotlib.colors import LogNorm

labels = {'LF': 'HMC-LF', 'AA': 'CHMC-AA', 'Newt': 'CHMC-Newt', 'Newton': 'CHMC-Newt', 'FPI': 'CHMC-FPI'}
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


def draw_heatmap(ax, values, accept, taus, tols, fmt, norm=None, fs=None, cbar_label=None, title=None):
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
    for (i, j), value in np.ndenumerate(values):
        if accept[i, j] < fail_rate:       # chain never moved: its wall time is not meaningful
            ax.text(j, i, "[x]", color="red", ha="center", va="center", fontweight="bold")
        elif np.isfinite(value):
            color = "black" if image.norm(value) > 0.5 else "white"
            ax.text(j, i, fmt(value), ha="center", va="center", color=color)
    return image


def plot_mean_runtime_heatmap(base, method, T=3.0, length=10_000, ppt=False):
    """Mean wall time per chain (s) over tau and tolerance, as in heatmap.ipynb."""
    fs = {'fontsize': 16} if ppt else {}
    taus, tols, seconds, accept = cell_arrays(load_cells(base, method, T, length))
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    image = draw_heatmap(ax, seconds, accept, taus, tols, lambda v: f"{v:.3g}", fs=fs,
                         title=f"{labels.get(method, method)}: mean runtime, {length:,} MCMC iterations, T={T:g}")
    fig.colorbar(image, ax=ax).set_label("Mean wall-clock time per chain (s)", **fs)
    return fig, ax, seconds, accept, taus, tols


def plot_runtime_ratio_heatmap(base, methods, baseline='LF', T=3.0, length=10_000, ppt=False,
                               divider=(0.125, 0.25), baseline_tau=0.0625):
    """Wall time of each method divided by HMC's at tau = baseline_tau (above 1 means slower).

    One reference time is used for every cell. HMC has no solver tolerance, so its time at
    baseline_tau is the mean over all its saved runs (every tolerance folder is a repeat).
    Cells where the method's chain never moved show [x].
    A red dotted line is drawn between the two tau values in `divider` (None for no line).
    """
    fs = {'fontsize': 16} if ppt else {}
    base_taus, _, base_seconds, _ = cell_arrays(load_cells(base, baseline, T, length))
    lf_time = np.nanmean(base_seconds[int(np.argmin(np.abs(base_taus - baseline_tau)))])   # one number
    panels = []
    for method in methods:
        taus, tols, seconds, accept = cell_arrays(load_cells(base, method, T, length))
        panels.append((method, seconds / lf_time, accept))
    finite = np.concatenate([r[np.isfinite(r)].ravel() for _, r, _ in panels])
    norm = LogNorm(vmin=finite.min(), vmax=finite.max())         # shared scale across methods
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels) + 1, 5),
                             layout="constrained", squeeze=False)
    for ax, (method, ratio, accept) in zip(axes[0], panels):
        image = draw_heatmap(ax, ratio, accept, taus, tols, lambda v: f"{v:.2g}×", norm=norm, fs=fs,
                             title=f"{labels.get(method, method)} / {labels.get(baseline, baseline)} ($\\tau = {baseline_tau:g}$)")
        if divider is not None:
            lo, hi = (int(np.argmin(np.abs(taus - tau))) for tau in divider)
            ax.axhline((lo + hi) / 2, color="red", linestyle=":", linewidth=2.5)
    fig.colorbar(image, ax=axes[0].tolist()).set_label(
        rf"Wall time relative to {labels.get(baseline, baseline)} at $\tau = {baseline_tau:g}$", **fs)
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

    others = [m for m in meta['methods'] if m not in ('LF', 'FPI')]   # FPI left out of the ratio figure
    for method in meta['methods']:
        fig, *_ = plot_mean_runtime_heatmap(args.output, method, T, length, args.ppt)
        fig.savefig(plotpath / f'runtime_{method}.png', dpi=200)
        plt.close(fig)
    fig, *_ = plot_runtime_ratio_heatmap(args.output, others, 'LF', T, length, args.ppt)
    fig.savefig(plotpath / 'runtime_ratio_vs_LF.png', dpi=200)
    plt.close(fig)
    print(f'Plots saved to {plotpath}')
