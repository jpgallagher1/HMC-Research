# John Gallagher
# Sept 8, 2026
# Script to test Neals Funnel
# 



import sys
from pathlib import Path
sys.path.insert(0, "/data/johngallagher/HMC-Research/CHMC")

SCRATCH = Path("/scratch/johngallagher")

base = SCRATCH/Path(f"neals_results")
base.mkdir(exist_ok=True)


import time
import json

# JAX
import jax
jax.config.update("jax_enable_x64", True)


from jax import jit, grad, vmap
import jax.random as jr
import jax.numpy as jnp
import jax.scipy as jsp

# Custom
from datatypes import QP, IntegratorConfig,  gen_configs
from target import neals_funnel_logpdf
from hamiltonian import logpdf_hamiltonian
from sampler import gen_chmc_kernel, gen_hmc_kernel, hmc_sampler, chmc_sampler
from databasing import write_tree

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('m', type = int)
args = parser.parse_args()


records = []

## Sampler parameters
key = jr.PRNGKey(1)
n_runs = 10
d=10

### time integration parameters
taus = 2**-jnp.linspace(1, 5, 5)
# taus = jnp.linspace(0.11, 0.08, 7) # extended sweep of sub region
Ts = jnp.linspace(1, 5, 5)
lens = jnp.logspace(2, 4, 9, base=10, dtype=int)


### Solver parameters
tol = 1e-3
max_iter = 10
m=args.m
methods = [f'AA_m={m}']
method_base = SCRATCH/base/methods[0]



H = logpdf_hamiltonian(neals_funnel_logpdf, None)
gradH = grad(H)

# (numtaus, numTs)
configs = gen_configs(taus, Ts, tol=tol, max_iter=max_iter, n_pts=6,
                      integrator='AA', gen_gauss=False, AA_beta=1, AA_m=m)  
                      
# (numtaus, numTs) 

for row in configs:
    print(' | '.join(f'τ={c.τ:g}, T={c.T:g}, N={c.N}' for c in row))



for l in range(len(lens)):
    for j in range(len(taus)):
        for k in range(len(Ts)):
            chmc = gen_chmc_kernel(H, configs[k][j])    
            scan_chmc = jax.jit(lambda init, xs: jax.lax.scan(chmc, init, xs))
            for i in range(n_runs):
                key = jr.PRNGKey(i)
                keya, keyb = jr.split(key)
                chain_keys = jr.split(keya, lens[l])

                qp0 = QP(jr.normal(keyb, shape=(2*d,)))
                init = [qp0, 1, False]

                # Warm up / compile
                compiled_scan = scan_chmc.lower(init, chain_keys).compile()

                # Timed run
                start = time.perf_counter()
                _, (qps, deltaHs, accepted) = scan_chmc(init, chain_keys)
                jax.block_until_ready(qps)
                elapsed = time.perf_counter() - start

                folder = (
                    base
                    / methods[0]
                    / f"tau_{taus[j]}"
                    / f"T_{Ts[k]}"
                    / f"len_{lens[l]}"
                )
                filename = folder / f"run_{i}.npz"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"run_{i}.npz"
                jnp.savez(
                    path,
                    q=jnp.asarray(qps.q),
                    deltaHs=jnp.asarray(deltaHs),
                    accepted=jnp.asarray(accepted),
                    runtime = jnp.float64(elapsed),
                )
                records.append({
                    "method": methods[0],
                    "tau": float(taus[j]),
                    "T": float(Ts[k]),
                    "length": int(lens[l]),
                    "run": int(i),
                    "path": str(path.relative_to(base)),
                })

with open(method_base / f"metadata_{methods[0]}.json", "w") as f:
    json.dump({"files": records}, f, indent=2)

write_tree(method_base, f'file_tree_{methods[0]}.txt')