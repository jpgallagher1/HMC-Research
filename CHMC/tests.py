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


class TestDatatypes:
    def test_qp_roundtrip(self):
        q = jnp.array([1.0, 2.0])
        p = jnp.array([3.0, 4.0])
        arr = QP(q=q, p=p).to_array()
        recovered = QP.from_array(arr)
        np.testing.assert_array_equal(recovered.q, q)
        np.testing.assert_array_equal(recovered.p, p)

    def test_qp_dim(self):
        q = jnp.ones(5)
        p = jnp.ones(5)
        assert QP(q=q, p=p).dim == 5

    def test_integrator_config_fields(self):
        cfg = IntegratorConfig(0.1, 1.0, 10)
        assert cfg.τ == 0.1
        assert cfg.T == 1.0
        assert cfg.N == 10
        assert cfg.tol == 1e-2
        assert cfg.max_iter == 3
