# John Gallagher
# Aug 9, 2026
# Script to test pgeneralized gauss in McGregor & Wan '26
# 



import sys
from pathlib import Path
sys.path.insert(0, "/data/johngallagher/HMC-Research/CHMC")

SCRATCH = Path("/scratch/johngallagher")


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
from datatypes import QP, IntegratorConfig
from hamiltonian import gen_p_gauss_hamiltonian
from sampler import gen_chmc_kernel, gen_hmc_kernel, hmc_sampler, chmc_sampler

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('p', type = int)
parser.add_argument('d', type = int)
args = parser.parse_args()


records = []

key = jr.PRNGKey(1)
key1, key2 = jr.split(key, 2)
qp0 = QP(jr.normal(key1, shape=(2,)))
qp1 = QP(jr.normal(key2, shape=(2,)))

p = args.p
d = args.d
n_runs = 10
m=4
methods = [f'AA_m={m}']
# taus = jnp.array([0.1, 0.09])
taus = 2**-jnp.linspace(1, 4, 4)
Ts = jnp.linspace(1, 5, 5)
taus_, Ts_ = jnp.meshgrid(taus, Ts, indexing='ij') # (numtaus, numTs)
Ns = (Ts_/taus_).astype(int) # (numtaus, numTs)
lens = jnp.logspace(2, 4, 9, base=10, dtype=int)

tol = 1e-3
max_iter = 10

H = gen_p_gauss_hamiltonian(p, None)
gradH = grad(H)

def gen_configs(taus, Ts, **kw):
    """
    One IntegratorConfig per (T, τ) pair, N = round(T/τ) so T = N*τ holds exactly.
    Returns nested list configs[i][j] ~ (Ts[i], taus[j]), shape (len(Ts), len(taus)).
    Extra kwargs (tol, max_iter, n_pts, ...) pass through to every config.
    """
    return [[IntegratorConfig(τ=float(τ), T=int(round(T/τ))*float(τ), N=int(round(T/τ)), **kw)
             for τ in taus] for T in Ts]

# (numtaus, numTs)
configs = gen_configs(taus, Ts, tol=tol, max_iter=max_iter, n_pts=6,
                      integrator='AA', gen_gauss=False, AA_beta=1, AA_m=m)  
                      
# (numtaus, numTs) 
# i, j = 2,3
# print(f'taus [{i}]: {taus[i]}, Ts [{j}]:{Ts[j]}')
# config = configs[2][3]   # T=1, τ=2⁻⁴: same as the old single config
for row in configs:
    print(' | '.join(f'τ={c.τ:g}, T={c.T:g}, N={c.N}' for c in row))



def write_tree(root: Path, outfile="file_tree.txt"):
    """
    Write an ASCII tree of the directory rooted at `root`.
    """

    root = Path(root)

    def tree(path: Path, prefix=""):
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))

        for i, entry in enumerate(entries):
            last = i == len(entries) - 1
            connector = "└── " if last else "├── "

            yield prefix + connector + entry.name

            if entry.is_dir():
                extension = "    " if last else "│   "
                yield from tree(entry, prefix + extension)

    with open(outfile, "w") as f:
        f.write(root.name + "/\n")
        for line in tree(root):
            f.write(line + "\n")

    print(f"Saved tree to {outfile}")

base = SCRATCH/Path(f"p={p}_d={d}_results")
base.mkdir(exist_ok=True)



for l in range(len(lens)):
    for j in range(len(taus)):
        for k in range(len(Ts)):
            for i in range(n_runs):
                key = jr.PRNGKey(i)
                chain_keys = jr.split(key, lens[l])
                qp0 = QP(jr.normal(key, shape=(2*d,)))
                chmc = gen_chmc_kernel(H, configs[k][j])
                scan_chmc = jax.jit(lambda init, xs: jax.lax.scan(chmc, init, xs))

                init = [qp0, 1, False]
                

                # Warm up / compile
                jax.block_until_ready(scan_chmc(init, chain_keys))

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
method_base = SCRATCH/base/methods[0]
with open(method_base / f"metadata_{methods[0]}.json", "w") as f:
    json.dump({"files": records}, f, indent=2)

write_tree(method_base, f'file_tree_{methods[0]}.txt')