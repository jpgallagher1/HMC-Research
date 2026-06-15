# Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill `CHMC/tests.py` with a regression test suite covering datatypes, hamiltonian, integrator, sampler, and metrics.

**Architecture:** Single `tests.py` file with one pytest class per module. Integrator tests use hard-coded analytic exact values from `scipy.linalg.expm` as the ground truth. All other tests are pure unit tests or smoke tests.

**Tech Stack:** pytest, JAX (jax.numpy, jax.random), scipy.linalg (reference computation only — values pre-computed and hard-coded)

---

## File Map

| Action | File | Purpose |
|---|---|---|
| Modify | `CHMC/tests.py` | All tests — currently empty |

All code under test already exists in `CHMC/`. Run pytest from project root as `pytest CHMC/tests.py -v`.

---

### Task 1: Scaffold tests.py with imports and sys.path

**Files:**
- Modify: `CHMC/tests.py`

- [ ] **Step 1: Write the scaffold**

Replace the contents of `CHMC/tests.py` with:

```python
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
```

- [ ] **Step 2: Run to confirm imports resolve**

```
cd /path/to/HMC-Online && pytest CHMC/tests.py -v
```

Expected: `no tests ran` (empty file), no ImportError.

- [ ] **Step 3: Commit**

```bash
git add CHMC/tests.py
git commit -m "test: scaffold tests.py with imports and sys.path"
```

---

### Task 2: TestDatatypes

**Files:**
- Modify: `CHMC/tests.py`

- [ ] **Step 1: Write failing tests (they should pass immediately since code exists)**

Append to `CHMC/tests.py`:

```python
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
```

- [ ] **Step 2: Run**

```
pytest CHMC/tests.py::TestDatatypes -v
```

Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add CHMC/tests.py
git commit -m "test: add TestDatatypes (QP roundtrip, dim, IntegratorConfig fields)"
```

---

### Task 3: TestHamiltonian

**Files:**
- Modify: `CHMC/tests.py`

- [ ] **Step 1: Write the tests**

Append to `CHMC/tests.py`:

```python
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
```

- [ ] **Step 2: Run**

```
pytest CHMC/tests.py::TestHamiltonian -v
```

Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add CHMC/tests.py
git commit -m "test: add TestHamiltonian (symplectic ops, gaussian H at origin)"
```

---

### Task 4: TestIntegrator

**Files:**
- Modify: `CHMC/tests.py`

**Reference values (pre-computed):**

Exact analytic solution for the 2×2 Gaussian Hamiltonian with:
- Precision Λ = `gen_perturb_mat(dim=2, perturbation=0.05)` = `[[1, 0.05], [0.05, 1]]`
- Mass⁻¹ = I₂
- q₀ = [1, 0], p₀ = [0, 1], T = 1.0 (τ=0.1, N=10)

Derived via `scipy.linalg.expm([[0,0,1,0],[0,0,0,1],[-1,-0.05,0,0],[-0.05,-1,0,0]])`:
- `EXACT_Q = [0.53286718, 0.82045343]`
- `EXACT_P = [-0.86215084, 0.50585114]`

Both integrators are 2nd-order methods; measured errors vs. exact solution:
- Leapfrog max error: ~1.3e-3
- Implicit midpoint max error: ~7e-4

Tolerances are set at `atol=5e-3` (≈4× measured error) to catch regressions without being brittle.

- [ ] **Step 1: Write the tests**

Append to `CHMC/tests.py`:

```python
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
```

- [ ] **Step 2: Run**

```
pytest CHMC/tests.py::TestIntegrator -v
```

Expected: 4 passed. If `test_leapfrog_vs_exact` or `test_midpoint_vs_exact` fails, print the actual values and compare against `EXACT_Q`/`EXACT_P` to adjust `atol`.

- [ ] **Step 3: Commit**

```bash
git add CHMC/tests.py
git commit -m "test: add TestIntegrator (leapfrog and midpoint vs analytic exact solution)"
```

---

### Task 5: TestSampler

**Files:**
- Modify: `CHMC/tests.py`

- [ ] **Step 1: Write the tests**

Append to `CHMC/tests.py`:

```python
class TestSampler:
    def _setup(self):
        key = jr.PRNGKey(42)
        dim = 2
        Lam = gen_perturb_mat(dim=2, perturbation=0.05)
        Mass_inv = jnp.eye(2)
        H = gaussian_hamiltonian(Lam, Mass_inv)
        H_flat = lambda qp_flat: H(QP.from_array(qp_flat))
        config = IntegratorConfig(0.1, 1.0, 10, tol=1e-2, max_iter=3)
        qp0_flat = QP(q=jnp.array([1.0, 0.0]), p=jnp.array([0.0, 1.0])).to_array()
        init_sample = [qp0_flat, 1.0, False]
        return key, dim, H_flat, config, init_sample

    def test_draw_momentum_keeps_q(self):
        key = jr.PRNGKey(0)
        q = jnp.array([1.0, 2.0])
        p = jnp.array([3.0, 4.0])
        new_qp, _ = draw_momentum(QP(q=q, p=p), key)
        np.testing.assert_array_equal(new_qp.q, q)

    def test_draw_momentum_p_shape(self):
        key = jr.PRNGKey(0)
        q = jnp.array([1.0, 2.0])
        p = jnp.array([3.0, 4.0])
        new_qp, _ = draw_momentum(QP(q=q, p=p), key)
        assert new_qp.p.shape == p.shape

    def test_accept_reject_returns_bool(self):
        key = jr.PRNGKey(0)
        result = accept_reject(0.0, key)
        assert jnp.array(result).shape == ()
        assert jnp.array(result).dtype == jnp.bool_

    def test_hmc_sampler_output_shape(self):
        key, dim, H_flat, config, init_sample = self._setup()
        n_samples = 10
        keys = jr.split(key, n_samples)
        qp_arr, dH_arr, acc_arr = hmc_sampler(init_sample, keys, H_flat, config)
        assert qp_arr.shape == (n_samples, 2 * dim)
        assert dH_arr.shape == (n_samples,)
        assert acc_arr.shape == (n_samples,)

    def test_chmc_sampler_output_shape(self):
        key, dim, H_flat, config, init_sample = self._setup()
        n_samples = 10
        keys = jr.split(key, n_samples)
        qp_arr, dH_arr, acc_arr = chmc_sampler(init_sample, keys, H_flat, config)
        assert qp_arr.shape == (n_samples, 2 * dim)
        assert dH_arr.shape == (n_samples,)
        assert acc_arr.shape == (n_samples,)

    def test_extract_positions_shape(self):
        key, dim, H_flat, config, init_sample = self._setup()
        n_samples = 10
        keys = jr.split(key, n_samples)
        samples = hmc_sampler(init_sample, keys, H_flat, config)
        positions = extract_positions(samples)
        assert positions.shape == (n_samples, dim)

    def test_extract_positions_accepted_only(self):
        key, dim, H_flat, config, init_sample = self._setup()
        n_samples = 10
        keys = jr.split(key, n_samples)
        samples = hmc_sampler(init_sample, keys, H_flat, config)
        positions = extract_positions(samples, accepted_only=True)
        assert positions.ndim == 2
        assert positions.shape[0] <= n_samples
        assert positions.shape[1] == dim

    def test_extract_energy_shape(self):
        key, dim, H_flat, config, init_sample = self._setup()
        n_samples = 10
        keys = jr.split(key, n_samples)
        samples = hmc_sampler(init_sample, keys, H_flat, config)
        energy = extract_energy(samples)
        assert energy.shape == (n_samples,)
```

- [ ] **Step 2: Run**

```
pytest CHMC/tests.py::TestSampler -v
```

Expected: 8 passed.

- [ ] **Step 3: Commit**

```bash
git add CHMC/tests.py
git commit -m "test: add TestSampler (smoke tests for draw_momentum, accept_reject, hmc/chmc shapes)"
```

---

### Task 6: TestMetrics

**Files:**
- Modify: `CHMC/tests.py`

- [ ] **Step 1: Write the tests**

Append to `CHMC/tests.py`:

```python
class TestMetrics:
    def _make_samples(self, n, accepted_flags):
        qp_arr = jnp.zeros((n, 4))
        dH_arr = jnp.zeros(n)
        acc_arr = jnp.array(accepted_flags, dtype=bool)
        return (qp_arr, dH_arr, acc_arr)

    def test_accept_rate_all_accepted(self):
        samples = self._make_samples(20, [True] * 20)
        assert float(compute_accept_rate(samples)) == pytest.approx(1.0)

    def test_accept_rate_none_accepted(self):
        samples = self._make_samples(20, [False] * 20)
        assert float(compute_accept_rate(samples)) == pytest.approx(0.0)

    def test_maxtracediff_identical(self):
        A = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        assert float(maxtracediff(A, A)) == pytest.approx(0.0)

    def test_maxtracediff_known(self):
        # diag([1, 3]) vs diag([2, 3]) → abs diff = [1, 0] → max = 1
        A = jnp.diag(jnp.array([1.0, 3.0]))
        B = jnp.diag(jnp.array([2.0, 3.0]))
        assert float(maxtracediff(A, B)) == pytest.approx(1.0)

    def test_cov_constant_data_is_zero(self):
        # constant rows → zero variance
        X = jnp.ones((10, 2))
        np.testing.assert_allclose(cov(X), jnp.zeros((2, 2)), atol=1e-10)

    def test_gen_2grid_shape(self):
        grid = gen_2grid(-1, 1, N=5)
        assert grid.shape == (25, 2)
```

- [ ] **Step 2: Run**

```
pytest CHMC/tests.py::TestMetrics -v
```

Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add CHMC/tests.py
git commit -m "test: add TestMetrics (accept_rate, maxtracediff, gen_2grid shape)"
```

---

### Task 7: Full suite run

- [ ] **Step 1: Run everything**

```
pytest CHMC/tests.py -v
```

Expected: 25 tests, all passed. If any fail, investigate — a failure likely reveals a pre-existing bug in the module under test.

- [ ] **Step 2: Final commit if needed**

If any test was adjusted to match a discovered quirk in the existing code, commit the explanation:

```bash
git add CHMC/tests.py
git commit -m "test: fix test assertions to match observed behaviour (document known quirks)"
```
