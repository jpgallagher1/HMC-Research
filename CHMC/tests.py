import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from datatypes import QP, IntegratorConfig
from hamiltonian import gaussian_hamiltonian, J_sym, qJ_sym, pJ_sym
from integrator import gen_leapfrog, gen_midptNewtonFPI, FPI
from sampler import draw_momentum, accept_reject, hmc_sampler, chmc_sampler, extract_positions, extract_energy
from metrics import compute_accept_rate, cov, maxtracediff, gen_2grid
from target import gen_perturb_mat
