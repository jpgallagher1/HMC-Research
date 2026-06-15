# Test Suite Design — CHMC/HMC Regression Tests

**Date:** 2026-05-22  
**File:** `CHMC/tests.py`  
**Framework:** pytest  
**Goal:** Regression safety — catch when refactoring breaks existing behaviour

---

## Structure

Single `CHMC/tests.py` file with one pytest class per module.

```
TestDatatypes
TestHamiltonian
TestIntegrator
TestSampler
TestMetrics
```

---

## TestDatatypes

Verify core data structures round-trip correctly and report consistent metadata.

- `test_qp_roundtrip`: `QP(q, p).to_array()` followed by `QP.from_array()` recovers the original `q` and `p` exactly.
- `test_qp_dim`: `QP.dim` returns `q.shape[0]`.
- `test_integrator_config_fields`: `IntegratorConfig(τ, T, N)` stores fields and defaults `tol=1e-2`, `max_iter=3`.

---

## TestHamiltonian

Verify symplectic operations produce the correct algebraic output, and that the Gaussian Hamiltonian evaluates correctly at a known point.

- `test_j_sym`: `J_sym(QP(q, p))` returns `QP(p, -q)`.
- `test_qj_sym`: `qJ_sym(QP(q, p))` returns `QP(p, zeros)`.
- `test_pj_sym`: `pJ_sym(QP(q, p))` returns `QP(zeros, -q)`.
- `test_gaussian_hamiltonian_at_origin`: `gaussian_hamiltonian(Λ, I)(QP(zeros, zeros)) == 0.0`.
- `test_gaussian_hamiltonian_returns_scalar`: output has shape `()`.

---

## TestIntegrator

Core tests. Both integrators are checked against the **exact analytic solution** for a 2×2 Gaussian Hamiltonian with a small perturbation precision matrix.

### Setup

| Parameter | Value |
|---|---|
| `dim` | 2 |
| Mass matrix `M⁻¹` | `jnp.eye(2)` |
| Precision `Λ` | `gen_perturb_mat(dim=2, perturbation=0.05)` → `[[1, 0.05], [0.05, 1]]` |
| Initial position `q₀` | `[1.0, 0.0]` |
| Initial momentum `p₀` | `[0.0, 1.0]` |
| Step size `τ` | `0.1` |
| Steps `N` | `10` |
| Final time `T` | `1.0` |

### Ground truth

The Gaussian Hamiltonian yields linear Hamiltonian equations:

```
d/dt [q, p] = [[0, I], [-Λ, 0]] [q, p]
```

Exact solution at `T=1.0`:

```python
M_sys = np.block([[np.zeros((2,2)), np.eye(2)],
                  [-np.array(Λ),   np.zeros((2,2))]])
z_exact = scipy.linalg.expm(M_sys) @ np.array([1., 0., 0., 1.])
expected_q = z_exact[:2]   # hard-coded as constants in the test
expected_p = z_exact[2:]
```

These values are computed once and stored as `jnp.array([...])` constants so the test is deterministic and has no scipy runtime dependency.

### Tests

- `test_leapfrog_vs_exact`: leapfrog output `q`, `p` match `expected_q`, `expected_p` within `atol=1e-3` (first-order method; some drift expected). Tolerance to be confirmed when exact values are computed during implementation.
- `test_midpoint_vs_exact`: implicit midpoint output matches within `atol=1e-6` (higher-order, energy-conserving; much tighter tolerance). Tolerance to be confirmed during implementation.
- `test_leapfrog_output_shape`: output flat array has same shape `(2*dim,)` as input.
- `test_midpoint_output_shape`: same.

---

## TestSampler

Smoke tests only — verify shapes, types, and basic invariants. No statistical moment checks.

- `test_draw_momentum_keeps_q`: `draw_momentum(qp, key).q` equals original `qp.q`.
- `test_draw_momentum_changes_p`: `draw_momentum(qp, key).p` shape equals `qp.p.shape`.
- `test_accept_reject_returns_bool`: return type is a JAX bool scalar.
- `test_hmc_sampler_output_shape`: with `n_samples=10`, output tuple `(qp_flat, delta_H, accepted)` has shapes `(10, 2*dim)`, `(10,)`, `(10,)`.
- `test_chmc_sampler_output_shape`: same as above.
- `test_extract_positions_shape`: `extract_positions(samples)` returns `(n_samples, dim)`.
- `test_extract_positions_accepted_only`: filtered output has `<= n_samples` rows.
- `test_extract_energy_shape`: `extract_energy(samples)` returns `(n_samples,)`.

---

## TestMetrics

- `test_accept_rate_all_accepted`: mock samples with all-True accepted → rate `== 1.0`.
- `test_accept_rate_none_accepted`: all-False → rate `== 0.0`.
- `test_maxtracediff_identical`: `maxtracediff(A, A) == 0.0`.
- `test_maxtracediff_known`: for diagonal matrices with known difference, result equals max absolute diagonal difference.
- `test_gen_2grid_shape`: `gen_2grid(min=-1, max=1, N=5)` returns shape `(25, 2)`.

---

## Test environment

- JAX 64-bit enabled: `jax.config.update("jax_enable_x64", True)` at module level in `tests.py` (before any JAX imports are used).
- All tests run from `CHMC/` directory (or `sys.path` adjusted to include `CHMC/`).
- No network, no file I/O, no plotting in any test.
- Expected to complete in under 30 seconds total.

---

## Out of scope

- Target distribution tests (`target.py`) — deferred at user request.
- Statistical moment checks (mean ≈ 0, cov ≈ Σ) — deferred; would require ~1000 samples and a slow-test marker.
- Energy conservation property tests — covered implicitly by integrator accuracy vs analytic solution.
