# sept 16, 2026
# plotting the w1 metrics as seen below. 
# data is computed in w1 metrics, and run with w1.sh script
# data is plotted via w1plots.sh
# data is located in 

import sys
from pathlib import Path
sys.path.insert(0, "/data/johngallagher/HMC-Research/CHMC")

import numpy as np
import matplotlib.pyplot as plt

SCRATCH = Path("/scratch/johngallagher")
base = SCRATCH/'neals_results'
plotpath = Path('/home/johngallagher/data/HMC-Research/plots/neals_results')
plotpath.mkdir(parents=True, exist_ok=True) 

methods = ['LF','FPI', 'Newton','AA_m=4','AA_m=3', 'AA_m=2']
nmtds = len(methods)

taus = 2**-np.linspace(1, 5, 5)
ntaus = len(taus)
Ts = np.linspace(1, 5, 5)
nTs = len(Ts)
lens = np.logspace(2, 4, 9, base=10, dtype=int)
lens_LF = np.logspace(2, 5, 9, base=10, dtype=int)
lens_FPI = np.logspace(2,4.7, 9, base = 10, dtype = int)
METHOD_LENS = {'LF': lens_LF, 'FPI': lens_FPI}  
nlens = len(lens)

# z = np.load(base / 'AA_m=2' / 'w1_AA_m=2.npz')
# z.files          # ['w1', 'runtime', 'taus', 'Ts', 'lens', 'runs', ...]
# z['w1'].shape    # (5, 5, 9, 10)
hmc_w1_metrics = {}
hmc_runtimes = {}
for meth in methods:
    with np.load(base / meth / f"w1_{meth}.npz") as z:
        hmc_w1_metrics[meth] = z['w1']
        hmc_runtimes[meth] = z['runtime']

plot_methods = ['AA_m=2', 'LF','FPI','Newton']    
markers = {
    "AA_m=2": "-*",
    "LF": "-o",
    "FPI": "-v",
    "Newton": "-^",
}

for j in range(len(taus)):
    for k in range(len(Ts)):
        for Cidx, meth in enumerate(plot_methods):
            color = f"C{Cidx}"
            marker = markers[meth]
            plt.loglog(hmc_runtimes[meth][j,k,:], hmc_w1_metrics[meth][j,k, :], marker, color=color, alpha=0.15)
            
            label = "Mean HMC-LF" if meth == "LF" else f"Mean CHMC-{meth}"
            avg_hmc = np.nanmean(hmc_w1_metrics[meth][j,k, :], axis=-1)
            avg_hmc_time = np.nanmean(hmc_runtimes[meth][j,k, :], axis=-1)
            plt.loglog(avg_hmc_time, avg_hmc, marker, color=color, label=label, alpha=1)


        title=f"W1 Err versus MCMC time Neals"      
        subtitle1 = f"\n $\\tau = $ {taus[j]}, $T = $ {Ts[k]}"
        subtitle2 = f"\n AA_window: 2, tol=1e-3, max_iter = 10, AVF_npts = 6"
        plt.title(title+subtitle1+subtitle2)

        plt.xlabel(r"Time (s)")
        plt.ylabel(r"Wasserstein $W_1$ Error")
        plt.grid(which="minor")
        plt.grid(which="major")

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        print(handles, labels)
        # Mean HMC-LF, Mean CHMC-FPI, Mean CHMC-AA, Mean CHMC-Newton
        # ['avg CHMC-AA', 'avg HMC-LF', 'avg CHMC-FPI', 'avg CHMC-Newton']
        order = [1,2,0,3]
        plt.legend([handles[i] for i in order], [labels[i] for i in order])

        file=f'neals_tau{taus[j]}_T{Ts[k]}_{methods[0]}_W1_time_loglog_DRAFT.png'
        plt.savefig(plotpath/file, dpi = 200)
        plt.close()