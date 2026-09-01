# John Gallagher
# Aug 10, 2026
# Goal run W1 metric for p-chi to computer 
import sys
from pathlib import Path
sys.path.insert(0, "/data/johngallagher/HMC-Research/CHMC")

SCRATCH = Path("/scratch/johngallagher")
base = SCRATCH/'gmm_results'
plotpath = Path('/home/johngallagher/data/HMC-Research/plots/gmm_results')
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
lens_LF = jnp.logspace(2, 5, 9, base=10, dtype=int)
lens_FPI = jnp.logspace(2,4.7, 9, base = 10, dtype = int)
nlens = len(lens)



#setting up w1 metric
n_pts = 10_000 # numer of target samples for POT
n_projections=1000
seednum = 0


def sample_gmm(key, n):
    k1, k2 = jr.split(key, 2)
    out = jr.normal(k1, shape=(n,))
        #  w1*(sd1*x + mu1) + (1-w1)*(sd2*x + mu2)
    z=jr.bernoulli(k2, p=0.75, shape=(n,))
    return (1-z)*(0.2*out+0.25) + z*(0.15*out-0.5)

xt = np.array(sample_gmm(key, n_pts))

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
                    if meth != 'LF' and meth != 'FPI':
                        result = load_result(base, meth, taus[j], Ts[k], lens[l], run=i)
                        hmc_chain_np = np.squeeze(result['q'])
                        hmc_runtimes[meth][j,k,l,i] = result['runtime']
                        try:
                                        # ntaus, nTs, nlens, n_runs
                            hmc_w1_metrics[meth][j, k, l, i] = ot.wasserstein_1d(
                                hmc_chain_np, xt, seed=seednum
                            )
                            # seednum +=1
                        except Exception:
                            hmc_w1_metrics[meth][j,k,l, i] = 1
                                    # number doesn't matter, just need it not to fail and it will be replaced later. 
                    if meth =='FPI':
                        # UGH I AM REPEATING MYSELF
                        for l in range(len(lens_FPI)):
                            result = load_result(base, meth, taus[j], Ts[k], lens_FPI[l], run=i)
                            hmc_chain_np = np.squeeze(result['q'])
                            hmc_runtimes[meth][j,k,l,i] = result['runtime']
                            try:
                                            # ntaus, nTs, nlens, n_runs
                                hmc_w1_metrics[meth][j, k, l, i] = ot.wasserstein_1d(
                                    hmc_chain_np, xt
                                )
                                seednum +=1
                            except Exception:
                                hmc_w1_metrics[meth][j,k,l, i] = jnp.nan    
                    else: 
                        # UGH I AM REPEATING MYSELF AGAIN
                        for l in range(len(lens_LF)):
    
                            result = load_result(base, meth, taus[j], Ts[k], lens_LF[l], run=i)
                            hmc_chain_np = np.squeeze(result['q'])
                            hmc_runtimes[meth][j,k,l,i] = result['runtime']
                            try:
                                            # ntaus, nTs, nlens, n_runs
                                hmc_w1_metrics[meth][j, k, l, i] = ot.wasserstein_1d(
                                    hmc_chain_np, xt
                                )
                                seednum +=1
                            except Exception:
                                hmc_w1_metrics[meth][j,k,l, i] = jnp.nan    

### RUNNING ON AA m=3
## July 20, 2026
## 


methods = ['AA_m=2', 'LF','FPI','Newton']
markers = {
    "AA_m=2": "-*",
    "LF": "-o",
    "FPI": "-v",
    "Newton": "-^",
}
# for k in range(nTs):
#    for j in range(ntaus):
for j in range(len(taus)):
    for k in range(len(Ts)):
        for Cidx, meth in enumerate(methods):
            color = f"C{Cidx}"
            marker = markers[meth]
            plt.loglog(hmc_runtimes[meth][j,k,:], hmc_w1_metrics[meth][j,k, :], marker, color=color, alpha=0.15)
            
            label = "Mean HMC-LF" if meth == "LF" else f"Mean CHMC-{meth}"
            avg_hmc = np.nanmean(hmc_w1_metrics[meth][j,k, :], axis=-1)
            avg_hmc_time = np.nanmean(hmc_runtimes[meth][j,k, :], axis=-1)
            plt.loglog(avg_hmc_time, avg_hmc, marker, color=color, label=label, alpha=1)
            # if meth == 'LF':
            #     avg_hmc = np.mean(hmc_w1_metrics[meth][j,k, :], axis=-1)
            #     avg_hmc_time = np.mean(hmc_runtimes[meth][j,k, :], axis=-1)
            #     plt.semilogy(avg_hmc_time, avg_hmc, "-o", color=color, label="avg HMC", alpha=1)
            # else:
            #     avg_chmc = np.mean(hmc_w1_metrics[meth][j,k, :], axis=-1)
            #     avg_chmc_time = np.mean(hmc_runtimes[meth][j,k, :], axis=-1)
                
            #     plt.semilogy(avg_chmc_time, avg_chmc, "-*", color=f"C{Cidx}", label=f"avg CHMC-{meth}", alpha=1)

        title=f"W1 Err versus MCMC time GMM"      
        subtitle1 = f"\n $\\tau = $ {taus[j]}, $T = $ {Ts[k]}"
        subtitle2 = f"\n AA_window: 2, tol=1e-3, max_iter = 10, AVF_npts = 6"
        plt.title(title+subtitle1+subtitle2)
        plt.rcParams['text.usetex'] = True

        # 2. Enforce Computer Modern for both standard text and math
        plt.rcParams['font.family'] = 'serif'
        # plt.rcParams['font.serif'] = ['Computer Modern Roman']
        # plt.title(title + subtitle1+subtitle2)
        # plt.title(title+subtitle1)
        plt.xlabel(r"Time (s)")
        plt.ylabel(r"Wasserstein $W_1$ Error")
        plt.grid(which="minor")
        plt.grid(which="major")

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        handles
        print(handles, labels)
        # Mean HMC-LF, Mean CHMC-FPI, Mean CHMC-AA, Mean CHMC-Newton
        # ['avg CHMC-AA', 'avg HMC-LF', 'avg CHMC-FPI', 'avg CHMC-Newton']
        order = [1,2,0,3]
        plt.legend([handles[i] for i in order], [labels[i] for i in order])
        # plt.legend()

        file=f'gmm_tau{taus[j]}_T{Ts[k]}_{methods[0]}_Prec_W1_time_loglog_DRAFT.png'
        plt.savefig(plotpath/file, dpi = 200)