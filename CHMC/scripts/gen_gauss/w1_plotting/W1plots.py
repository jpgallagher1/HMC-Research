# John Gallagher
# Aug 10, 2026
# Goal run W1 metric for p-chi to computer 
import sys
from pathlib import Path
sys.path.insert(0, "/data/johngallagher/HMC-Research/CHMC")

SCRATCH = Path("/scratch/johngallagher")
base = SCRATCH/'p=4_d=100_results'
plotpath = Path('/home/johngallagher/data/HMC-Research/plots/w1_gen_gauss')
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

methods = ['LF','FPI','Newton','AA_m=4','AA_m=3', 'AA_m=2']
nmtds = len(methods)
# taus = jnp.array([0.1, 0.09])
# taus = 2**-jnp.linspace(1, 4, 4)
taus = [0.5, 0.25, 0.125, 0.0625]
ntaus = len(taus)

Ts = jnp.linspace(1, 5, 5)
nTs = len(Ts)
# taus_, Ts_ = jnp.meshgrid(taus, Ts, indexing='ij') # (numtaus, numTs)
# Ns = (Ts_/taus_).astype(int) # (numtaus, numTs)
lens = jnp.logspace(2, 4, 9, base=10, dtype=int)
nlens = len(lens)



#setting up w1 metric
n_pts = 10_000 # numer of target samples for POT
n_projections=1000
seednum = 0

mu_s = np.zeros(shape=(d,))
# for setup in wasserstein metric
a, b = np.ones((n_pts,)) / n_pts, np.ones((n_pts,)) / n_pts  # uniform distribution on samples

def sample_gen_gauss(key, num_sampl=n_pts, p_exp=p, dim=d):
    return jr.generalized_normal(key, p_exp, shape=(num_sampl, dim))


xt = np.array(sample_gen_gauss(key, num_sampl=n_pts, p_exp=p, dim=d))

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
    for j in range(len(taus)):
        for k in range(len(Ts)):
            for l in range(len(lens)):
                for i in range(n_runs):
                    result = load_result(base, meth, taus[j], Ts[k], lens[l], run=i)
                    hmc_chain_np = np.squeeze(result['q'])
                    hmc_runtimes[meth][j,k,l,i] = result['runtime']
                    try:
                                    # ntaus, nTs, nlens, n_runs
                        hmc_w1_metrics[meth][j, k, l, i] = ot.sliced_wasserstein_distance(
                            hmc_chain_np, xt, a=None, b=None, n_projections=n_projections, p=1, seed=seednum
                        )
                        # seednum +=1
                    except Exception:
                        hmc_w1_metrics[meth][j,k,l, i] = 1
                                    # number doesn't matter, just need it not to fail and it will be replaced later. 

# for some reason doesn't want to automatically compute so i just ran through and actually grabbed the entries once. 
# for meth in methods:
#     for j in range(len(taus)):
#         for k in range(len(Ts)):
#             for l in range(len(lens)):
#                 for i in range(n_runs):
#                     hmc_w1_metrics[meth][j,k,l,i]
# incorporated in the above forloop now. 
# hmc_runtimes = {meth: np.zeros(shape=(ntaus, nTs, nlens, n_runs)) for meth in methods}
# for meth in methods:
#     for j in range(len(taus)):
#         for k in range(len(Ts)):
#             for l in range(len(lens)):
#                 for i in range(n_runs):
#                     result = load_result(base, meth, taus[j], Ts[k], lens[l], run=i)
#                     hmc_runtimes[meth][j,k,l,i] = result['runtime']



    # for j in range(len(taus)):
    #     for k in range(len(Ts)):
    #         for l in range(len(lens)):
    #             for i in range(n_runs):

# j = -1

# chmc_metrics = hmc_w1_metrics['AA_m=2']
# hmc_metrics = hmc_w1_metrics['LF']

def gen_w1_pchi_plots_iters(
        lens,
        chmc_metrics,
        hmc_metrics,
        meth,
        j,
        k,
        d: int,
        p: int,
        avg=True,
    ):
        title=f"W1 Err versus MCMC Iterations p-chi: {p}, ddof: {d}"  
        plt.loglog(lens, chmc_metrics[j,k, :], "-*", color="C0", alpha=0.15)
        plt.loglog(lens, hmc_metrics[j, k, :], "-o", color="C1", alpha=0.15)

        subtitle1 = f"\n $\\tau = $ {taus[j]}, $T = $ {Ts[k]}"
        if avg:
            avg_chmc = np.mean(chmc_metrics[j,k, :], axis=-1)
            plt.loglog(lens, avg_chmc, "-*", color="C0", label=f"avg CHMC-{meth}", alpha=1)
            avg_hmc = np.mean(hmc_metrics[j,k, :], axis=-1)
            plt.loglog(lens, avg_hmc, "-o", color="C1", label="avg HMC", alpha=1)
        else:
            plt.title(title + subtitle1)
        plt.title(title+subtitle1)
        plt.xlabel("MCMC Iterations")
        plt.ylabel("Error")
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.legend()
# gen_w1_pchi_plots(lens, hmc_w1_metrics['AA'], hmc_w1_metrics['LF'] , 'AA' j=-1, k=2, avg = True, d=d, p=p)


for meth in ['FPI','Newton','AA_m=4','AA_m=3', 'AA_m=2']:
    fig, axes = plt.subplots(ntaus, nTs, figsize=(30, 20),squeeze=False)
    fig.subplots_adjust(
        left=0.04,
        right=0.98,
        bottom=0.06,
        top=0.94,
        wspace=0.35,
        hspace=0.45,
    )
    axes = axes.flatten()
    for k in range(nTs):
        for j in range(ntaus):
            plt.sca(axes[j*nTs+k])
            gen_w1_pchi_plots_iters(lens, hmc_w1_metrics[meth], hmc_w1_metrics['LF'], meth, j=j, k=k, avg = True, d=d, p=p)
    file=f'pgauss_p{p}_d{d}_tau_T_{meth}_vsLF_W1_iter.png'
    plt.savefig(plotpath/file)
    plt.close(fig)