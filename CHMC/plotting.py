"""
Description:
    Plotting Functiosn for Metrics
    USE THE CORRECT ENVIRONMENT:  HMC-Research

Author: John Gallagher
Created: 2026-02-27
Last Modified: 2026-02-27
Version: 0.1
"""
import numpy as np
from matplotlib import pyplot as plt


def gen_plots(lens, 
              chmc_metrics, 
              hmc_metrics, 
              config, 
              i, 
              dims: int,
              conds: list,
              slope=False, 
              avg=True
              ):
    plt.loglog(lens, chmc_metrics[i,:], '-*', color = 'C0', alpha = 0.15)
    plt.loglog(lens, hmc_metrics[i,:], '-o', color = 'C1', alpha = 0.15)
    title = 'Cov Max Diag Err versus chainlength'
    subtitle1 = f'\n$dim$ = {dims[0]}, $\kappa$ = {conds[i]: 0.2f}, $\\tau = $ {config.τ}, $N = $ {config.N}'
    if avg:
        avg_chmc = np.mean(chmc_metrics[i,:], axis = 1)
        plt.loglog(lens, avg_chmc, '-*', color = 'C0', label ='avg CHMC', alpha = 1)
        avg_hmc = np.mean(hmc_metrics[i,:], axis = 1)
        plt.loglog(lens, avg_hmc , '-o', color = 'C1', label ='avg HMC', alpha = 1)
    if slope:
        chmc_p = np.polyfit(np.log(lens), np.log(avg_chmc), 1)
        hmc_p = np.polyfit(np.log(lens), np.log(avg_hmc), 1)

        # slop of avg chain subtitle
        subtitle2 = f'\n CHMC avg. slope = {chmc_p[0]:.2f}, HMC avg. slope = {hmc_p[0]:.2f}'

        # combined subtitle
        plt.title(title+subtitle1+subtitle2)
    else:
        plt.title(title + subtitle1)
    plt.xlabel('MCMC Iterations')
    plt.ylabel('Error')
    plt.grid(which='minor')
    plt.grid(which='major')
    plt.legend()

def gen_τ_plots(
    lens,
    chmc_metrics,
    hmc_metrics,
    config,
    i,
    dims: int,
    cond: list,
    kappa=True,
    dim=False,
    slope=False,
    avg=True,
    title="Cov Max Diag Err versus MCMC Iterations",
):
    plt.loglog(lens, chmc_metrics[i, :], "-*", color="C0", alpha=0.15)
    plt.loglog(lens, hmc_metrics[i, :], "-o", color="C1", alpha=0.15)
    if kappa:
        subtitle1 = f"\n$dim$ = {dims[0]}, $\\kappa$ = {cond: 0.2f}, $\\tau = $ {config.τ}, $N = $ {config.N}"
    if dim and not kappa:
        subtitle1 = f"\n$dim$ = {dims[i]}, $\\tau = $ {config.τ}, $N = $ {config.N}"
    if avg:
        avg_chmc = np.mean(chmc_metrics[i, :], axis=-1)
        plt.loglog(lens, avg_chmc, "-*", color="C0", label="avg CHMC", alpha=1)
        avg_hmc = np.mean(hmc_metrics[i, :], axis=-1)
        plt.loglog(lens, avg_hmc, "-o", color="C1", label="avg HMC", alpha=1)
    if slope:
        chmc_p = np.polyfit(np.log(lens), np.log(avg_chmc), 1)
        hmc_p = np.polyfit(np.log(lens), np.log(avg_hmc), 1)
        subtitle2 = f"\n CHMC avg. slope = {chmc_p[0]:.2f}, HMC avg. slope = {hmc_p[0]:.2f}"
        plt.title(title + subtitle1 + subtitle2)
    else:
        plt.title(title + subtitle1)
    plt.xlabel("MCMC Iterations")
    plt.ylabel("Error")
    plt.grid(which="minor")
    plt.grid(which="major")
    plt.legend()
