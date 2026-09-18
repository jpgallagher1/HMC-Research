# John Gallagher
# Sept 17, 2026
# Visualizing bias in Neal's funnel generated samples
# Comparison of generated samples versus pdf level sets, for first two coordinates.

# Boilerplate preamble for importing raw data. Copy pasted from neals/w1metrics.py
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("tau_idx", type=int, default=-1)
parser.add_argument("T_idx", type=int, default=0)

args = parser.parse_args()

import sys
from pathlib import Path
sys.path.insert(0, "/data/johngallagher/HMC-Research/CHMC")

SCRATCH = Path("/scratch/johngallagher")
base = SCRATCH/'neals_results'
plotpath = Path('/home/johngallagher/data/HMC-Research/plots/neals_results/scatter')
plotpath.mkdir(parents=True, exist_ok=True)

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jr

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


key = jr.PRNGKey(-1)

run = 0

methods = ['LF', 'FPI', 'Newton', 'AA_m=2', 'AA_m=3', 'AA_m=4']

taus = 2**-jnp.linspace(1, 5, 5)
tau = taus[args.tau_idx]

Ts = jnp.linspace(1, 5, 5)
TT = Ts[args.T_idx]

lens = jnp.logspace(2, 4, 9, base=10, dtype=int)
lens_LF = jnp.logspace(2, 5, 9, base=10, dtype=int)
lens_FPI = jnp.logspace(2, 4.7, 9, base=10, dtype=int)
METHOD_LENS = {'LF': lens_LF, 'FPI': lens_FPI}

LOG_2PI = np.log(2 * np.pi)
MASSES = (0.5, 0.9, 0.99)
LABEL = {"LF": "HMC-LF"}


def log_normal(x, sd):
    return -0.5 * (x / sd) ** 2 - np.log(sd) - 0.5 * LOG_2PI


def logp_yy_x1(yy, x1):
    """(yy, x_1) marginal of the funnel: x_2 ... x_9 integrate out."""
    return log_normal(yy, 3.0) + log_normal(x1, np.exp(0.5 * yy))


def hpd_levels(logp_at_samples, masses=MASSES):
    """log-density thresholds t_α with P(log p ≥ t_α) = α; returned ascending for contour."""
    levels = {float(np.quantile(logp_at_samples, 1 - a)): a for a in masses}
    return sorted(levels), levels


def load_result(base, method, tau, T, length, run):
    """
    navigating the file path generated from the forloops.
    """
    path = (
        Path(base)
        / method
        / f"tau_{float(tau):.12g}"
        / f"T_{float(T):.1f}"
        / f"len_{int(length)}"
        / f"run_{run}.npz"
    )
    with np.load(path) as f:
        return np.asarray(f["q"])


chains = {mm: load_result(base, mm, tau, TT, METHOD_LENS.get(mm, lens)[-1], run) for mm in methods}

# Level sets of the exact (yy, x_1) marginal: contours holding 50, 90, 99 % of its mass.
rng = np.random.default_rng(0)
yy_s = 3.0 * rng.standard_normal(1_000_000)
x_s = np.exp(0.5 * yy_s) * rng.standard_normal(1_000_000)
lv_x, lab_x = hpd_levels(logp_yy_x1(yy_s, x_s))

ymin = min(-10.0, min(q[:, 0].min() for q in chains.values())) - 1.0
ymax = max(10.0, max(q[:, 0].max() for q in chains.values())) + 1.0
xmax = np.quantile(np.abs(x_s), 0.99)          # target 99 % of |x_1|; excursions clipped
yy_g = np.linspace(ymin, ymax, 600)
x_g = np.linspace(-xmax, xmax, 600)
Xg, Yx = np.meshgrid(x_g, yy_g)
LPx = logp_yy_x1(Yx, Xg)

fig, axes = plt.subplots(2, 3, figsize=(4.4 * 3, 9.5), sharex=True, sharey=True)
flat = axes.ravel()
for cc, meth in enumerate(methods):
    yy, x1 = chains[meth][:, 0], chains[meth][:, 1]
    LL = yy.size

    ax = flat[cc]
    cs = ax.contour(Xg, Yx, LPx, levels=lv_x, colors="0.35", linewidths=0.8)
    ax.clabel(cs, fmt={l: f"{100 * lab_x[l]:g}%" for l in lv_x}, fontsize=7)
    ax.scatter(x1, yy, s=2, alpha=0.3, rasterized=True)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel("yy")
    ax.set_title(f"{LABEL.get(meth, 'CHMC-' + meth)}  (len {LL}, run {run})\n"
                 f"mean yy {yy.mean():.2f},  sd yy {yy.std():.2f}")

fig.suptitle(f"Neal's funnel samples vs target level sets,  $\\tau$ = {tau:g}, $T$ = {TT:g}")
fig.tight_layout()
file = f"neals_scatter_tau{tau:g}_T{TT:g}_run{run}.png"
fig.savefig(plotpath / file, dpi=200)
print(f"wrote {plotpath / file}")