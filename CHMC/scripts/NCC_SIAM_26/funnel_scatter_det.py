# John Gallagher
# Oct 26, 2026
# Visualizing bias in Neal's funnel generated samples, with and without the log|det J| correction.
# Two 1x3 figures (HMC-LF, CHMC-AA, CHMC-Newton) of (x_1, yy) samples against the 1 and 2 sd
# level sets of the target: one from the determinant-adjusted run (timed_blocks_funnel_det.py) and one from the
# unadjusted run (timed_blocks_funnel.py). Leapfrog is volume preserving, so its panel is the
# same algorithm in both figures (different seeds and folders).
# Plotting only: reads the saved block files and computes nothing else.

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pinnacles locations, as in neals/funnel_scatter.py; override with the flags to run locally.
SCRATCH = Path("/scratch/johngallagher")
PLOTS = Path("/home/johngallagher/data/HMC-Research/plots")

parser = argparse.ArgumentParser()
parser.add_argument("--det", type=Path, default=SCRATCH/"neals_funnel_det_tau0.0625_T3")
parser.add_argument("--nodet", type=Path, default=SCRATCH/"neals_funnel_blocks")
parser.add_argument("--tau", type=float, default=0.0625)
parser.add_argument("--T", type=float, default=3.0)
parser.add_argument("--run", type=int, default=0)
parser.add_argument("--n_samples", type=int, default=10_000)   # same length for every method; LF/FPI are extended past it
parser.add_argument("--sds", type=float, nargs="+", default=[1, 2])   # level sets drawn, in standard deviations
parser.add_argument("--plotpath", type=Path, default=PLOTS/"neals_funnel_det"/"scatter")
args = parser.parse_args()

plotpath = args.plotpath
plotpath.mkdir(parents=True, exist_ok=True)

methods = ["LF", "AA", "Newt"]
LABEL = {"LF": "HMC-LF", "AA": "CHMC-AA", "Newt": "CHMC-Newton", "Newton": "CHMC-Newton"}
COLOR = {"AA": "C0", "LF": "C1", "FPI": "C2", "Newt": "C3", "Newton": "C3"}   # as in the p-chi DRAFT plot

SIGMA = 3.0          # yy ~ N(0, SIGMA^2)


def sd_radius2(yy, x1):
    """
    Squared distance in standard deviations: (yy/σ)² + (x_1/e^{yy/2})².
    yy is k sd from 0 at |yy| = kσ, and x_1 | yy is k sd from 0 at |x_1| = k e^{yy/2},
    so the level set sd_radius2 = k² spans yy in [-kσ, kσ] and is k sd wide in x_1 at yy = 0.
    """
    return (yy / SIGMA) ** 2 + x1 ** 2 * np.exp(-yy)


def load_chain(base, method, tau, T, run, n_samples):
    """
    First n_samples of one chain, concatenating block_00000000.npz, block_00000001.npz, ...
    from base/method/tau_*/T_*/run_*.
    """
    folder = Path(base) / method / f"tau_{tau}" / f"T_{T}" / f"run_{run}"
    qs, accs = [], []
    for path in sorted(folder.glob("block_*.npz")):
        with np.load(path) as f:
            qs.append(f["q"])
            accs.append(f["accepted"])
        if sum(len(q) for q in qs) >= n_samples:
            break
    if not qs:
        raise FileNotFoundError(f"No blocks in {folder}")
    return np.concatenate(qs)[:n_samples], np.concatenate(accs)[:n_samples]


runs = {"det": (args.det, r"with $\log|\det J|$ correction"),
        "nodet": (args.nodet, r"without $\log|\det J|$ correction")}
chains = {(rr, mm): load_chain(base, mm, args.tau, args.T, args.run, args.n_samples)
          for rr, (base, _) in runs.items() for mm in methods}

# Same axes in both figures so they can be compared side by side.
ymin = min(-10.0, min(q[:, 0].min() for q, _ in chains.values())) - 1.0
ymax = max(10.0, max(q[:, 0].max() for q, _ in chains.values())) + 1.0
kmax = max(args.sds)
yy_k = np.linspace(-kmax * SIGMA, kmax * SIGMA, 2001)
xmax = 1.5 * np.max(np.sqrt(np.clip(kmax**2 - (yy_k / SIGMA)**2, 0, None)) * np.exp(yy_k / 2))
# 1.5 x the widest point of the outer level set; excursions past it are clipped
yy_g = np.linspace(ymin, ymax, 600)
x_g = np.linspace(-xmax, xmax, 600)
Xg, Yx = np.meshgrid(x_g, yy_g)
R2 = sd_radius2(Yx, Xg)
lv_x = sorted(k ** 2 for k in args.sds)

for rr, (base, desc) in runs.items():
    fig, axes = plt.subplots(1, 3, figsize=(4.4 * 3, 5.2), sharex=True, sharey=True)
    for cc, meth in enumerate(methods):
        q, acc = chains[rr, meth]
        yy, x1 = q[:, 0], q[:, 1]

        ax = axes[cc]
        cs = ax.contour(Xg, Yx, R2, levels=lv_x, colors="0.35", linewidths=0.8)
        ax.clabel(cs, fmt={l: f"{np.sqrt(l):g} sd" for l in lv_x}, fontsize=7)
        ax.scatter(x1, yy, s=2, alpha=0.3, color=COLOR[meth], rasterized=True)
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"$x_1$")
        if cc == 0:
            ax.set_ylabel("yy")
        ax.set_title(f"{LABEL[meth]}  (len {yy.size}, run {args.run})\n"
                     f"mean yy {yy.mean():.2f},  sd yy {yy.std():.2f},  min yy {yy.min():.2f}\n"
                     f"acceptance {acc.mean():.2f}", fontsize=9)

    fig.suptitle(f"Neal's funnel samples vs 1, 2 sd level sets, {desc},  "
                 f"$\\tau$ = {args.tau:g}, $T$ = {args.T:g}")
    fig.tight_layout()
    file = f"neals_scatter_{rr}_tau{args.tau:g}_T{args.T:g}_run{args.run}.png"
    fig.savefig(plotpath / file, dpi=200)
    plt.close(fig)
    print(f"wrote {plotpath / file}")
