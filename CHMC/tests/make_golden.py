# June 15, 2026
# Making an update to the QP object for easier handling throughout the code 

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
from jax import jit, grad, jacfwd
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from datatypes import QP, IntegratorConfig
from integrator import gen_leapfrog, gen_midptNewtonFPI
from target import gen_2D_perturb_vals, gen_perturb_mat
from hamiltonian import gaussian_hamiltonian
from sampler import hmc_sampler, chmc_sampler, extract_positions

# ====
# Config
# ====
jax.config.update("jax_enable_x64", True)
golden_path = 'CHMC/tests/data/'
lf_traj_str = 'lf_traj'
midptNewton_traj_str = 'midptNewton_traj'
hmc_samples_str = 'hmc_samples'
chmc_samples_str = 'chmc_samples'
# config_str = 'config'
tests = [lf_traj_str, midptNewton_traj_str, hmc_samples_str, chmc_samples_str]

key = jr.PRNGKey(1)
dim = 2
runs = 1

#integrator
tau = 0.2
N = 5
T = N*tau
tol = 1e-2
max_iter = 2

#sampler
mcmc_iter = 4

#target distribution
κ = 2

# ====
# Derived quantities
# ====

# Integrator
config = IntegratorConfig(τ=tau, T=T, N=N, tol=tol, max_iter=max_iter)

# Hamiltonian
perturb_val = gen_2D_perturb_vals(κ)
mass_inv = jnp.eye(dim)

cov_mat = gen_perturb_mat(dim,perturb_val)
prec_mat = jnp.linalg.inv(cov_mat)

H = gaussian_hamiltonian(prec_mat, mass_inv)
H_flat = lambda x_flat, H=H: H(QP.from_array(x_flat))
gradH_flat = grad(H_flat)

# ====
# Integrator setup
# ==== 

lf = gen_leapfrog(gradH_flat, config)
midptNewton= gen_midptNewtonFPI(gradH_flat, config)

# ====
# PRNG Keys
# ====

qp0_key, hmc0_key, chmc0_key = jr.split(key, 3)

hmc_qp_init_key, chmc_qp_init_key = jr.split(qp0_key, 2)
hmc_keys = jr.split(hmc0_key, mcmc_iter)
chmc_keys = jr.split(chmc0_key, mcmc_iter)

# ====
# Jit-compile samplers
# ====

jhmc_sampler = jit(hmc_sampler, static_argnums=(2,3))
jchmc_sampler = jit(chmc_sampler, static_argnums=(2,3,4))


# ====
# Initialize states
# ====

# initial state
qp_init = jr.normal(qp0_key, shape=(2*dim))

#initial sampler state
init = [qp_init, 1, False]


# ====
# Integrate
# ====

lf_traj = lf(qp_init)
midptNewton_traj = midptNewton(qp_init)
print(lf_traj)
print(midptNewton_traj)

# ====
# Sample
# ====
hmc_chain = extract_positions(jhmc_sampler(init, hmc_keys, H_flat, config))
chmc_chain = extract_positions(jchmc_sampler(init, chmc_keys, H_flat, config))

hmc_samples = np.array(hmc_chain)
chmc_samples = np.array(chmc_chain)

print(f'Saving {lf_traj_str}@ {golden_path+lf_traj_str}')
np.save(golden_path+lf_traj_str, lf_traj)    
print(f'Saving {midptNewton_traj_str}@ {golden_path+midptNewton_traj_str}')
np.save(golden_path+midptNewton_traj_str, midptNewton_traj)    
print(f'Saving {hmc_samples_str}@ {golden_path+hmc_samples_str}')
np.save(golden_path+hmc_samples_str, hmc_samples)    
print(f'Saving {chmc_samples_str}@ {golden_path+chmc_samples_str}')
np.save(golden_path+chmc_samples_str, chmc_samples)