# John Gallagher
# Sept 14, 2026
# Script to test Neals Funnel
# 
# I need to stop repeating myself, like in the preamble of scratch/Path. 



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('T', type=float, choices=[1, 2, 3, 4, 5],
                    help='final integration time')
parser.add_argument('tau_idx', type=int, choices=range(5),
                    help='index into taus')
parser.add_argument('len_idx', type=int, choices=range(9),
                    help='index into lens')
parser.add_argument('run_idx', type=int, nargs='?', default=None, choices=range(10),
                    help='optional: run only this run index')
args = parser.parse_args()

import sys
from pathlib import Path
sys.path.insert(0, "/data/johngallagher/HMC-Research/CHMC")

SCRATCH = Path("/scratch/johngallagher")

base = SCRATCH/Path(f"neals_results")
base.mkdir(parents=True, exist_ok=True)


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




records = []

## Sampler parameters
key = jr.PRNGKey(1)
n_runs = 10
d=10
 
### time integration parameters
taus = 2**-jnp.linspace(1, 5, 5)
T = args.T
lens = jnp.logspace(2, 4, 9, base=10, dtype=int)
 
j = args.tau_idx
l = args.len_idx
run_ids = range(n_runs) if args.run_idx is None else [args.run_idx]
 
 
### Solver parameters
tol = 1e-3
max_iter = 10
methods = [f'Newton']
method_base = base/methods[0]
 
 
 
H = logpdf_hamiltonian(neals_funnel_logpdf, None)
gradH = grad(H)
 
# One configuration per tau for the supplied T.
configs = gen_configs(taus, [T], tol=tol, max_iter=max_iter, n_pts=6,
                      integrator='AVF_NewtonFPI_T', gen_gauss=False, AA_beta=1)[0]
                      
 
print(' | '.join(f'τ={c.τ:g}, T={c.T:g}, N={c.N}' for c in configs))
 
 
chmc = gen_chmc_kernel(H, configs[j])
scan_chmc = jax.jit(lambda init, xs: jax.lax.scan(chmc, init, xs))
for i in run_ids:
    key = jr.PRNGKey(i)
    keya, keyb = jr.split(key)
    chain_keys = jr.split(keya, lens[l])
 
    qp0 = QP(jr.normal(keyb, shape=(2*d,)))
    init = [qp0, 1, False]
 
    # Finish input preparation before compiling or timing.
    jax.block_until_ready((init, chain_keys))
 
    # Compile once per (length, tau, T). No execution warm-up.
    if i == run_ids[0]:
        print(f"Compiling T={T}, tau={taus[j]}, length={lens[l]}", flush=True)
        compiled_scan = scan_chmc.lower(init, chain_keys).compile()
 
    # Time this chain only; wait for the entire result.
    print(f"Starting T={T}, tau={taus[j]}, length={lens[l]}, run={i}", flush=True)
    start = time.perf_counter()
    result = compiled_scan(init, chain_keys)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start
    _, (qps, deltaHs, accepted) = result
 
    folder = (
        base
        / methods[0]
        / f"tau_{taus[j]}"
        / f"T_{T}"
        / f"len_{lens[l]}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"run_{i}.npz"
    jnp.savez(
        path,
        q=jnp.asarray(qps.q),
        deltaHs=jnp.asarray(deltaHs),
        accepted=jnp.asarray(accepted),
        runtime = jnp.float64(elapsed),
    )
    print(f"Saved {path}: runtime={elapsed:.6f}s", flush=True)
    records.append({
        "method": methods[0],
        "tau": float(taus[j]),
        "T": float(T),
        "length": int(lens[l]),
        "run": int(i),
        "path": str(path.relative_to(base)),
    })

# Not running this because it's cumbersome.  
# One metadata file per job, so parallel jobs don't overwrite each other.
# suffix = "" if args.run_idx is None else f"_run_{args.run_idx}"
# with open(method_base / f"metadata_{methods[0]}_T_{T}_tau_{j}_len_{l}{suffix}.json", "w") as f:
    # json.dump({"files": records}, f, indent=2)
 