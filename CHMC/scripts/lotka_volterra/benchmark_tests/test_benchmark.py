"""Focused numerical and experimental-design checks, separate from legacy CHMC tests."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import benchmark_ess as bench
from ess_diagnostics import diagnose, efficiency
from lv_model import build_model, load_data, DEFAULT_DATA
from datatypes import QP, IntegratorConfig
from integrator import gen_AVF_AA_T, gen_leapfrog


def test_ess_distinguishes_independence_from_autocorrelation():
    rng = np.random.default_rng(13)
    iid = rng.normal(size=(4, 2000, 2))
    correlated = np.zeros_like(iid)
    correlated[:, 0] = iid[:, 0]
    for i in range(1, iid.shape[1]):
        correlated[:, i] = .9*correlated[:, i-1] + np.sqrt(1-.9**2)*iid[:, i]
    a, b = diagnose(iid), diagnose(correlated)
    assert np.min(a['ess_bulk']) > 5000
    assert np.max(a['rhat']) < 1.01
    assert np.max(b['ess_bulk']) < np.min(a['ess_bulk'])/5
    assert np.all(np.isfinite(b['ess_tail']))


def test_stuck_and_shifted_chains_cannot_win():
    stuck = np.ones((4, 100, 2))
    assert not efficiency(stuck, 1.)['converged']
    rng = np.random.default_rng(17)
    shifted = rng.normal(size=(4, 1000, 2)) + np.arange(4)[:, None, None]*3
    result = efficiency(shifted, 1.)
    assert not result['converged']
    assert np.isnan(result['seconds_per_min_bulk_ess'])
    shifted[0, 0, 0] = np.nan
    assert np.isnan(diagnose(shifted)['ess_bulk'][0])


def test_folded_rhat_detects_scale_disagreement():
    x = np.random.default_rng(8).normal(size=(4, 2000, 1))
    x[0] *= 8
    assert diagnose(x)['rhat'][0] > 1.1


def test_timing_objective_includes_requested_costs():
    x = np.random.default_rng(2).normal(size=(4, 2000, 2))
    r = efficiency(x, 10., 20.)
    assert r['converged']
    assert r['total_seconds_per_min_bulk_ess'] == pytest.approx(3*r['seconds_per_min_bulk_ess'])
    with pytest.raises(ValueError): diagnose(x[0])
    with pytest.raises(ValueError): efficiency(x, 0.)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_aa_matches_exact_quadratic_avf_step(dtype):
    with jax.enable_x64(dtype==jnp.float64):
        initial = QP(jnp.array([.7, -.2, .3, .8], dtype=dtype))
        H = lambda qp: .5*jnp.sum(qp.x**2)
        config = IntegratorConfig(τ=.1, T=.1, N=1, integrator='AA', AA_m=2,
                                  n_pts=2, max_iter=30, tol=1e-6)
        result = jax.jit(gen_AVF_AA_T(jax.grad(H), config))(initial)
        J = np.block([[np.zeros((2,2)),np.eye(2)],[-np.eye(2),np.zeros((2,2))]])
        exact = np.linalg.solve(np.eye(4)-.05*J, (np.eye(4)+.05*J)@np.asarray(initial.x))
        np.testing.assert_allclose(result.x, exact, atol=2e-6)
        assert abs(float(H(result)-H(initial))) < 2e-6


@pytest.mark.parametrize('method', ['HMC', 'AA'])
def test_kernel_uses_distinct_momentum_and_acceptance_keys(method):
    H = lambda qp: .5*jnp.sum(qp.x**2)
    config = IntegratorConfig(.2, .2, 1, integrator='AA', AA_m=2, n_pts=2, max_iter=20, tol=1e-5)
    initial = QP(jnp.array([.5, -.1, 0., 0.], dtype=jnp.float32))
    key = jr.PRNGKey(12); pk, ak = jr.split(key)
    qp0 = QP.from_qp(initial.q, jr.normal(pk, initial.q.shape, dtype=initial.x.dtype))
    integrator = gen_AVF_AA_T(jax.grad(H), config) if method=='AA' else gen_leapfrog(jax.grad(H), config)
    proposed = integrator(qp0)
    delta = H(qp0)-H(proposed)
    accept = bool(jnp.log(jr.uniform(ak)) <= jnp.minimum(0., delta))
    state, (_, actual_delta, actual_accept, finite) = jax.jit(bench.make_kernel(H, config, method))(initial, key)
    assert bool(actual_accept)==accept and bool(finite)
    np.testing.assert_allclose(actual_delta, delta, atol=1e-6)
    np.testing.assert_allclose(state.x, proposed.x if accept else qp0.x, atol=1e-6)


def test_nonfinite_proposal_is_rejected():
    H = lambda qp: jnp.where(jnp.all(jnp.abs(qp.q)<1), .5*jnp.sum(qp.x**2), jnp.nan)
    initial = QP(jnp.array([.5, 0.], dtype=jnp.float32))
    state, (_, delta, accepted, finite) = jax.jit(bench.make_kernel(H, IntegratorConfig(10.,10.,1), 'HMC'))(initial,jr.PRNGKey(1))
    assert not bool(accepted) and not bool(finite)
    np.testing.assert_array_equal(state.q, initial.q)


def test_grid_preserves_actual_T_equals_N_tau():
    args = bench.parser().parse_args(['--times','.25','--taus','.04'])
    grid = bench.configurations(args)
    for s in grid:
        assert s['T']==s['N']*s['tau'] and s['N']>=1
        assert s['requested_T']==.25
    assert {s['method'] for s in grid}=={'HMC','AA'}


def test_gaussian_end_to_end_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'build_model', lambda *a: (lambda x:-.5*jnp.sum(x*x), None,jnp.zeros(7)))
    args = bench.parser().parse_args(['--preset','smoke','--output',str(tmp_path),'--chains','2','--draws','8'])
    args.warmup=0
    bench.run(args)
    results = list(tmp_path.glob('*/result.json'))
    assert len(results)==2
    assert all(json.loads(p.read_text())['eligible'] is False for p in results)
    assert json.loads((tmp_path/'comparison.json').read_text())['AA_speedup_over_HMC'] is None
    mtimes = [p.stat().st_mtime_ns for p in results]
    bench.run(args)
    assert mtimes==[p.stat().st_mtime_ns for p in results]
    args.ode_dt=.01
    with pytest.raises(ValueError, match='manifest differs'): bench.run(args)


def test_short_unconverged_results_are_not_ranked(tmp_path):
    for method in ('AA','HMC'):
        directory = tmp_path/method; directory.mkdir()
        bench.atomic_json(directory/'result.json', dict(configuration={'method':method},
                          eligible=False, seconds_per_min_bulk_ess=.001, diagnostics={}))
    assert bench.summarize(tmp_path)['best']=={}


def test_data_and_model_gradient_against_finite_differences():
    ts, hare, lynx = load_data()
    assert len(ts)==21 and ts[0]==0 and ts[-1]==20
    with jax.enable_x64(True):
        posterior, predict, center = build_model(ode_dt=.02)
        fn, derivative = jax.jit(posterior), jax.jit(jax.grad(posterior))
        value, gradient = jax.block_until_ready((fn(center), derivative(center)))
        assert np.isfinite(value) and np.all(np.isfinite(gradient))
        # Two directions exercise dynamics and likelihood scale independently.
        for index in (0,6):
            direction = jnp.eye(7)[index]; h=1e-5
            difference = float((fn(center+h*direction)-fn(center-h*direction))/(2*h))
            assert float(gradient[index]) == pytest.approx(difference, rel=2e-4, abs=2e-3)
        prey, predator = predict(center)
        assert np.all(np.asarray(prey)>0) and np.all(np.asarray(predator)>0)


def test_ode_refinement_at_initial_center():
    with jax.enable_x64(True):
        coarse, _, center = build_model(ode_dt=.02)
        fine, _, _ = build_model(ode_dt=.01)
        assert abs(float(jax.jit(coarse)(center)-jax.jit(fine)(center))) < .02


def test_comparison_reports_correct_speedup_and_repeats(tmp_path):
    for method, seconds in [('HMC', .4), ('AA', .2)]:
        directory = tmp_path/method; directory.mkdir()
        bench.atomic_json(directory/'result.json', dict(configuration={'method':method},
            eligible=True, seconds_per_min_bulk_ess=seconds,
            diagnostics={'per_repeat':[{'seconds_per_min_bulk_ess':seconds}]}))
    result = bench.summarize(tmp_path)
    assert result['AA_speedup_over_HMC'] == pytest.approx(2.)
    assert result['paired_repeat_speedups'] == [2.]
