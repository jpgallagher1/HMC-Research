
"""
John Gallagher
Sept 15, 2026
Goal run W1 metric for Neals on pinnacles

Description:
    Sliced W1 between stored Neal's funnel chains and an exact reference sample.
    One file per method: <base>/<meth>/w1_<meth>.npz. Plotting is in plot_w1.py.
    USE THE CORRECT ENVIRONMENT:  HMC-Research
 
Usage:
    python -u w1_by_method.py Newton        # one method (one SLURM job per method)
    python -u w1_by_method.py               # all methods in `methods`, serially
 
Output arrays, indexed [j, k, l, i] = [τ, T, chain length, run]:
    w1, runtime            (ntaus, nTs, nlens, n_runs)
    taus, Ts, lens, runs   coordinates
    n_pts, n_projections


access via: 
z = np.load(base / 'AA_m=2' / 'w1_AA_m=2.npz')
z.files          # ['w1', 'runtime', 'taus', 'Ts', 'lens', 'runs', ...]
z['w1'].shape    # (5, 5, 9, 10)
hmc_w1_metrics = {}
for meth in methods:
    with np.load(base / meth / f"w1_{meth}.npz") as z:
        hmc_w1_metrics[meth] = z['w1']
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('meth', type=str, choices=['LF','FPI', 'Newton','AA_m=4','AA_m=3', 'AA_m=2'], help='integrator type')
args = parser.parse_args()

import sys
from pathlib import Path
sys.path.insert(0, "/data/johngallagher/HMC-Research/CHMC")

SCRATCH = Path("/scratch/johngallagher")
base = SCRATCH/'neals_results'
plotpath = Path('/home/johngallagher/data/HMC-Research/plots/neals_results')
plotpath.mkdir(parents=True, exist_ok=True) 

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax import jit, grad, vmap

import numpy as np

import metrics
import matplotlib.pyplot as plt
import ot

from plotting import gen_τ_plots

key = jr.PRNGKey(-1)

p = 4
d = 100

n_runs = 10
runs = jnp.arange(0, n_runs)

methods = [args.meth]
nmtds = len(methods)
# taus = jnp.array([0.1, 0.09])
taus = 2**-jnp.linspace(1, 5, 5)
# taus = [0.5, 0.25, 0.125, 0.0625]
# taus = jnp.linspace(0.11, 0.08, 7)
ntaus = len(taus)

Ts = jnp.linspace(1, 5, 5)
nTs = len(Ts)
# taus_, Ts_ = jnp.meshgrid(taus, Ts, indexing='ij') # (numtaus, numTs)
# Ns = (Ts_/taus_).astype(int) # (numtaus, numTs)
lens = jnp.logspace(2, 4, 9, base=10, dtype=int)
lens_LF = jnp.logspace(2, 5, 9, base=10, dtype=int)
lens_FPI = jnp.logspace(2,4.7, 9, base = 10, dtype = int)
METHOD_LENS = {'LF': lens_LF, 'FPI': lens_FPI}  
nlens = len(lens)



#setting up w1 metric
n_pts = 20_000 # numer of target samples for POT
n_projections=1000
seednum = 0


def sample_neals(key, n, dim=10):
    k1, k2 = jr.split(key, 2)
    ys = 3*jr.normal(k1, shape=(n,1))
        #  w1*(sd1*x + mu1) + (1-w1)*(sd2*x + mu2)
    zs=jnp.exp(ys/2)*jr.normal(k2,shape=(n,dim-1))
    return jnp.concatenate([ys,zs], axis = -1)

xt = np.array(sample_neals(key, n_pts))

# for l in range(len(lens)):
#     for j in range(len(taus)):
#         for k in range(len(Ts)):
#             for i in range(n_runs):


def load_result(base, method, tau, T, length, run):
    """
    navigating the file path generated from the forloops. 
    result = load_result(
        base,
        method = "AA",
        tau = 2**-1,
        T = 1.0,
        length = 1000,
        run = 3,
    )

    q = result["q"]
    deltaHs = result["deltaHs"]
    accepted = result["accepted"]
    runtime = result["runtime"]

    """
    path = (
        Path(base)
        / method
        / f"tau_{float(tau):.12g}"
        / f"T_{float(T):.1f}"
        / f"len_{int(length)}"
        / f"run_{run}.npz"
    )
    return jnp.load(path)

# try to keep the same ordering as the file tree so you don't have to think about it as much 
        # / method
        # / f"tau_{float(tau):.12g}"
        # / f"T_{float(T):.1f}"
        # / f"len_{int(length)}"
        # / f"run_{run}.npz"


hmc_w1_metrics = {meth: np.zeros(shape=(ntaus, nTs, nlens, n_runs)) for meth in methods}

hmc_runtimes = {meth: np.zeros(shape=(ntaus, nTs, nlens, n_runs)) for meth in methods}


for meth in methods:
    method_lens = METHOD_LENS.get(meth, lens)
    for j in range(len(taus)):
        for k in range(len(Ts)):
            for i in range(n_runs):
                for l in range(len(method_lens)):
                    result = load_result(base, meth, taus[j], Ts[k], method_lens[l], run=i)
                    hmc_chain_np = np.squeeze(result['q'])
                    hmc_runtimes[meth][j,k,l,i] = result['runtime']
                    try:
                                    # ntaus, nTs, nlens, n_runs
                        hmc_w1_metrics[meth][j, k, l, i] = ot.sliced_wasserstein_distance(
                        hmc_chain_np, xt, a=None, b=None, n_projections=n_projections, p=1, seed=seednum
                        )
                        seednum +=1
                    except Exception:
                        hmc_w1_metrics[meth][j,k,l, i] = 1
                                # number doesn't matter, just need it not to fail and it will be replaced later. 
            print(f"{meth} tau={float(taus[j]):g} T={float(Ts[k]):g} done", flush=True)
filepath = base / meth / f"w1_{meth}.npz"
np.savez(
    filepath,
    w1=hmc_w1_metrics[meth], runtime=hmc_runtimes[meth],
    taus=np.asarray(taus), Ts=np.asarray(Ts), lens=np.asarray(method_lens), runs=np.arange(n_runs),
    n_pts=n_pts, n_projections=n_projections, ref_seed=-1,
)
print(f"wrote {filepath}", flush=True)

