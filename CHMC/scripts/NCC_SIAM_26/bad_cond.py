# Oct 18, 2026
# Need to adapt old code to work on timed blocks of 200? 
# incorporate CHMC-AA, CHMC-N_K, CHMC_FPI.
# example for data generation: CHMC/Sandbox/AVF/pchi_sampler_AA.ipynb, CHMC/Sandbox/AVF/pchi_sampler_FPI.ipynb, CHMC/Sandbox/AVF/pchi_sampler_LF.ipynb, CHMC/Sandbox/AVF/pchi_sampler_Newton.ipynb
# plot plots in new style like p-chi plots
# CHMC/Sandbox/AVF/W1plots-p=4_d=400_m=2_centered.ipynb



### RUNNING ON AA m=3
## July 20, 2026
## 


# methods = ['AA', 'LF','FPI','Newton']
# markers = {
#     "AA": "-*",
#     "LF": "-o",
#     "FPI": "-v",
#     "Newton": "-^",
# }
# # for k in range(nTs):
# #    for j in range(ntaus):
# j = -3
# k = -2
# for Cidx, meth in enumerate(methods):
#     color = f"C{Cidx}"
#     marker = markers[meth]
#     plt.loglog(hmc_runtimes[meth][j,k,:], hmc_w1_metrics[meth][j,k, :], marker, color=color, alpha=0.15)
    
#     label = "Mean HMC-LF" if meth == "LF" else f"Mean CHMC-{meth}"
#     avg_hmc = np.nanmean(hmc_w1_metrics[meth][j,k, :], axis=-1)
#     avg_hmc_time = np.nanmean(hmc_runtimes[meth][j,k, :], axis=-1)
#     plt.loglog(avg_hmc_time, avg_hmc, marker, color=color, label=label, alpha=1)
#     # if meth == 'LF':
#     #     avg_hmc = np.mean(hmc_w1_metrics[meth][j,k, :], axis=-1)
#     #     avg_hmc_time = np.mean(hmc_runtimes[meth][j,k, :], axis=-1)
#     #     plt.semilogy(avg_hmc_time, avg_hmc, "-o", color=color, label="avg HMC", alpha=1)
#     # else:
#     #     avg_chmc = np.mean(hmc_w1_metrics[meth][j,k, :], axis=-1)
#     #     avg_chmc_time = np.mean(hmc_runtimes[meth][j,k, :], axis=-1)
        
#     #     plt.semilogy(avg_chmc_time, avg_chmc, "-*", color=f"C{Cidx}", label=f"avg CHMC-{meth}", alpha=1)

# title=f"W1 Err versus MCMC time p-chi: {p}, ddof: {d}"      
# subtitle1 = f"\n $\\tau = $ {taus[j]}, $T = $ {Ts[k]}"
# subtitle2 = f"\n AA_window: 3, tol=1e-3, max_iter = 10, AVF_npts = 6"

# plt.rcParams['text.usetex'] = True

# # 2. Enforce Computer Modern for both standard text and math
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Computer Modern Roman']
# plt.title(title + subtitle1+subtitle2)
# plt.title(title+subtitle1)
# plt.xlabel(r"Time (s)")
# plt.ylabel(r"Wasserstein $W_1$ Error")
# plt.grid(which="minor")
# plt.grid(which="major")

# ax = plt.gca()
# handles, labels = ax.get_legend_handles_labels()
# handles
# print(handles, labels)
# # Mean HMC-LF, Mean CHMC-FPI, Mean CHMC-AA, Mean CHMC-Newton
# # ['avg CHMC-AA', 'avg HMC-LF', 'avg CHMC-FPI', 'avg CHMC-Newton']
# order = [1,2,0,3]
# plt.legend([handles[i] for i in order], [labels[i] for i in order])
# # plt.legend()

# file=f'pchi_p{p}_d{d}_tau{taus[j]}_T{Ts[k]}_{methods[0]}_Prec_W1_time_loglog_DRAFT.png'
# plt.savefig(plotpath/file, dpi = 200)

