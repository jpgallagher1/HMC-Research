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


class TestHamiltonian:
    def test_j_sym(self):
        q = jnp.array([1.0, 2.0])
        p = jnp.array([3.0, 4.0])
        result = J_sym(QP(q=q, p=p))
        np.testing.assert_array_equal(result.q, p)
        np.testing.assert_array_equal(result.p, -q)

    def test_qj_sym(self):
        q = jnp.array([1.0, 2.0])
        p = jnp.array([3.0, 4.0])
        result = qJ_sym(QP(q=q, p=p))
        np.testing.assert_array_equal(result.q, p)
        np.testing.assert_array_equal(result.p, jnp.zeros_like(q))

    def test_pj_sym(self):
        q = jnp.array([1.0, 2.0])
        p = jnp.array([3.0, 4.0])
        result = pJ_sym(QP(q=q, p=p))
        np.testing.assert_array_equal(result.q, jnp.zeros_like(p))
        np.testing.assert_array_equal(result.p, -q)

    def test_gaussian_hamiltonian_at_origin(self):
        Lam = jnp.eye(2)
        Mass_inv = jnp.eye(2)
        H = gaussian_hamiltonian(Lam, Mass_inv)
        qp = QP(q=jnp.zeros(2), p=jnp.zeros(2))
        assert float(H(qp)) == pytest.approx(0.0)

    def test_gaussian_hamiltonian_returns_scalar(self):
        Lam = jnp.eye(2)
        Mass_inv = jnp.eye(2)
        H = gaussian_hamiltonian(Lam, Mass_inv)
        qp = QP(q=jnp.ones(2), p=jnp.ones(2))
        result = H(qp)
        assert jnp.array(result).shape == ()


class TestIntegrator:
    # Analytic exact solution — computed via scipy.linalg.expm on the linear ODE system
    # d/dt [q,p] = [[0,I],[-Lam,0]] [q,p], Lam = gen_perturb_mat(2, 0.05), T=1.0
    EXACT_Q = jnp.array([0.53286718, 0.82045343])
    EXACT_P = jnp.array([-0.86215084, 0.50585114])

    def _setup(self):
        Lam = gen_perturb_mat(dim=2, perturbation=0.05)
        Mass_inv = jnp.eye(2)
        H = gaussian_hamiltonian(Lam, Mass_inv)
        gradH_flat = jax.grad(lambda qp_flat: H(QP.from_array(qp_flat)))
        q0 = jnp.array([1.0, 0.0])
        p0 = jnp.array([0.0, 1.0])
        qp0_flat = QP(q=q0, p=p0).to_array()
        config = IntegratorConfig(0.1, 1.0, 10, tol=1e-8, max_iter=20)
        return gradH_flat, qp0_flat, config

    def test_leapfrog_vs_exact(self):
        gradH_flat, qp0_flat, config = self._setup()
        integrator = gen_leapfrog(gradH_flat, config)
        qp_out = QP.from_array(integrator(qp0_flat))
        np.testing.assert_allclose(qp_out.q, self.EXACT_Q, atol=5e-3)
        np.testing.assert_allclose(qp_out.p, self.EXACT_P, atol=5e-3)

    def test_midpoint_vs_exact(self):
        gradH_flat, qp0_flat, config = self._setup()
        integrator = gen_midptNewtonFPI(gradH_flat, config)
        qp_out = QP.from_array(integrator(qp0_flat))
        np.testing.assert_allclose(qp_out.q, self.EXACT_Q, atol=5e-3)
        np.testing.assert_allclose(qp_out.p, self.EXACT_P, atol=5e-3)

    def test_leapfrog_output_shape(self):
        gradH_flat, qp0_flat, config = self._setup()
        integrator = gen_leapfrog(gradH_flat, config)
        out = integrator(qp0_flat)
        assert out.shape == qp0_flat.shape

    def test_midpoint_output_shape(self):
        gradH_flat, qp0_flat, config = self._setup()
        integrator = gen_midptNewtonFPI(gradH_flat, config)
        out = integrator(qp0_flat)
        assert out.shape == qp0_flat.shape
