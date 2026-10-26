#!/usr/bin/env python
"""Restartable AA versus HMC seconds/ESS experiment; see BENCHMARK.md."""
import argparse
import csv
import hashlib
import itertools
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxopt import AndersonAcceleration
from datatypes import QP, IntegratorConfig
from integrator import gen_AVF, gen_AVF_AA_T, gen_leapfrog
from hamiltonian import logpdf_hamiltonian
from lv_model import build_model, DEFAULT_DATA, PARAMETERS
from ess_diagnostics import efficiency


def configurations(args):
    if args.preset == 'smoke':
        taus, times, histories, nodes, tols = [.01], [.01], [2], [2], [1e-3]
    else:
        taus, times = args.taus, args.times
        histories, nodes, tols = args.histories, args.nodes, args.tols
    configs = []
    for method in args.methods:
        extras = [(0, 0, 0.)] if method == 'HMC' else itertools.product(histories, nodes, tols)
        for m, p, tol in extras:
            for tau, requested_t in itertools.product(taus, times):
                n = max(1, int(round(requested_t/tau)))
                configs.append(dict(method=method, tau=tau, T=n*tau,
                                    requested_T=requested_t, N=n, m=m, nodes=p, tol=tol,
                                    max_iter=args.max_iter, beta=args.beta))
    return configs


def make_config(spec):
    return IntegratorConfig(τ=spec['tau'], T=spec['T'], N=spec['N'],
                            integrator='AA' if spec['method']=='AA' else 'LF',
                            AA_m=max(1, spec['m']), n_pts=max(1, spec['nodes']),
                            tol=spec['tol'], max_iter=spec['max_iter'], AA_beta=spec['beta'])


def make_kernel(H, config, method):
    """Existing numerical integrators; independent momentum and MH randomness."""
    gradH = jax.grad(H)
    integrator = gen_AVF_AA_T(gradH, config) if method == 'AA' else gen_leapfrog(gradH, config)
    def kernel(qp, key):
        momentum_key, accept_key = jr.split(key)
        qp0 = QP.from_qp(qp.q, jr.normal(momentum_key, qp.q.shape, dtype=qp.x.dtype))
        proposal = integrator(qp0)
        delta = H(qp0)-H(proposal)
        finite = jnp.all(jnp.isfinite(proposal.x)) & jnp.isfinite(delta)
        accepted = finite & (jnp.log(jr.uniform(accept_key, dtype=qp.x.dtype)) <= jnp.minimum(0., delta))
        state = QP(jnp.where(accepted, proposal.x, qp0.x))
        return state, (state.q, delta, accepted, finite)
    return kernel


def probe(H, config, method, initial, key):
    """Untimed full trajectory: actual fixed-point residuals and iteration counts."""
    start = QP.from_qp(initial.q, jr.normal(key, initial.q.shape, dtype=initial.x.dtype))
    if method != 'AA':
        return {}
    avf = gen_AVF(jax.grad(H), config)
    solver = AndersonAcceleration(lambda y, x: avf(x, y), history_size=config.AA_m,
                                  beta=config.AA_beta, tol=config.tol, maxiter=config.max_iter)
    def step(x, _):
        out = solver.run(avf(x, x), x)
        residual = jnp.linalg.norm(avf(x, out.params).x-out.params.x)
        return out.params, (out.state.iter_num, out.state.error, residual)
    _, (iters, errors, residuals) = jax.block_until_ready(
        jax.jit(lambda x: jax.lax.scan(step, x, None, length=config.N))(start))
    i, e, r = map(np.asarray, (iters, errors, residuals))
    return dict(probe_mean_iterations=float(i.mean()), probe_cap_fraction=float((i>=config.max_iter).mean()),
                probe_nonfinite_steps=int((~np.isfinite(r)).sum()),
                probe_unconverged_steps=int(((r>config.tol) | ~np.isfinite(r)).sum()),
                probe_max_residual=float(np.max(r)), probe_max_reported_error=float(np.max(e)),
                nominal_gradient_evaluations=int(config.n_pts*(config.N+i.sum())))


def serializable(value):
    if isinstance(value, dict): return {k:serializable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [serializable(v) for v in value]
    if isinstance(value, np.ndarray): return serializable(value.tolist())
    if isinstance(value, np.generic): return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def atomic_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(serializable(value), indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def save_npz(path, **arrays):
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def summarize(root):
    records = [json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))]
    rows = []
    for r in records:
        row = {**r['configuration'], **{k:v for k,v in r.items() if k not in ('configuration','diagnostics','probe')}}
        row.update(r.get('probe', {}))
        rows.append(row)
    if rows:
        fields = sorted(set().union(*(r.keys() for r in rows)))
        with (root/'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    best = {}
    for method in ('HMC','AA'):
        eligible = [r for r in records if r['configuration']['method']==method and r.get('eligible')
                    and r.get('seconds_per_min_bulk_ess') is not None]
        if eligible:
            best[method] = min(eligible, key=lambda r:r['seconds_per_min_bulk_ess'])
    report = {'best': best, 'AA_speedup_over_HMC': None,
              'interpretation': 'Performance of these implementations; no target-invariance claim.'}
    if len(best)==2:
        report['AA_speedup_over_HMC'] = best['HMC']['seconds_per_min_bulk_ess']/best['AA']['seconds_per_min_bulk_ess']
        report['paired_repeat_speedups'] = [h['seconds_per_min_bulk_ess']/a['seconds_per_min_bulk_ess']
            for h, a in zip(best['HMC']['diagnostics'].get('per_repeat', []),
                            best['AA']['diagnostics'].get('per_repeat', []))]
        hd, ad = best['HMC']['diagnostics'], best['AA']['diagnostics']
        if all(k in hd and k in ad for k in ('mean_theta','sd_theta','ess_mean_theta')):
            standard_error = np.sqrt(np.array(hd['sd_theta'])**2/np.array(hd['ess_mean_theta']) +
                                     np.array(ad['sd_theta'])**2/np.array(ad['ess_mean_theta']))
            report['posterior_mean_difference_in_mcse_units'] = (
                np.abs(np.array(hd['mean_theta'])-np.array(ad['mean_theta']))/standard_error).tolist()

    atomic_json(root/'comparison.json', report)
    return report


def run(args):
    jax.config.update('jax_enable_x64', args.dtype=='float64')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    model_settings = dict(ode_dt=args.ode_dt, notebook_priors=args.notebook_priors,
                          data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest())
    manifest = dict(schema=1, model=model_settings, dtype=args.dtype, chains=args.chains,
                    warmup=args.warmup, draws=args.draws, repeats=args.repeats, seed=args.seed,
                    init_scale=args.init_scale, chunk=args.chunk, probe=args.probe, rhat_limit=args.rhat_limit, min_ess=args.min_ess,
                    implementation_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in
                        [Path(__file__), HERE/'lv_model.py', HERE/'ess_diagnostics.py',
                         HERE.parents[1]/'integrator.py', HERE.parents[1]/'datatypes.py'])).hexdigest(),
                    package_versions={name:version(name) for name in ('jax','jaxopt','diffrax','numpy','scipy','numpyro')}, devices=[str(d) for d in jax.devices()])
    manifest_path = root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Output manifest differs; use a new output directory')
    atomic_json(manifest_path, manifest)
    posterior, _, center = build_model(args.data, args.ode_dt, args.notebook_priors)
    H = logpdf_hamiltonian(posterior, None)
    dtype = jnp.float64 if args.dtype=='float64' else jnp.float32
    center = center.astype(dtype)
    energy_gradient = jax.jit(lambda x: (H(x), jax.grad(H)(x)))
    for spec in configurations(args):
        identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
        folder = root/(spec['method']+'_'+identity); folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            print('SKIP', folder.name, flush=True); continue
        config = make_config(spec); kernel = make_kernel(H, config, spec['method'])
        # One compiled transition avoids enormous nested scan compilations and allows checkpoints.
        example = QP.from_qp(center, jnp.zeros_like(center))
        t0 = time.perf_counter()
        compiled = jax.jit(kernel).lower(example, jr.PRNGKey(0)).compile()
        compile_seconds = time.perf_counter()-t0
        jax.block_until_ready(compiled(example, jr.PRNGKey(999)))  # untimed execution warm-up
        print('RUN', folder.name, spec, flush=True)
        allq, alldelta, allaccepted, allfinite = [], [], [], []
        sampling_seconds = warmup_seconds = 0.
        first_rng = jr.fold_in(jr.fold_in(jr.PRNGKey(args.seed), 0), 0)
        first_init_key, _ = jr.split(first_rng)
        initial = QP.from_qp(center+args.init_scale*jr.normal(first_init_key, center.shape, dtype=dtype), jnp.zeros_like(center))
        for repeat in range(args.repeats):
            for chain in range(args.chains):
                checkpoint = folder/f'chain_{repeat}_{chain}.npz'
                if checkpoint.exists():
                    with np.load(checkpoint) as saved:
                        allq.append(saved['q']); alldelta.append(saved['deltaH'])
                        allaccepted.append(saved['accepted']); allfinite.append(saved['finite'])
                        sampling_seconds += float(saved['sampling_seconds']); warmup_seconds += float(saved['warmup_seconds'])
                    print('  SKIP', checkpoint.name, flush=True); continue
                rng = jr.fold_in(jr.fold_in(jr.PRNGKey(args.seed), repeat), chain)
                init_key, run_key = jr.split(rng)
                q = center+args.init_scale*jr.normal(init_key, center.shape, dtype=dtype)
                state = QP.from_qp(q, jnp.zeros_like(q))
                energy, gradient = jax.block_until_ready(energy_gradient(state))
                if not np.isfinite(energy) or not np.all(np.isfinite(gradient.x)):
                    raise ValueError(f'Nonfinite initial energy/gradient for repeat {repeat}, chain {chain}')
                keys = jr.split(run_key, args.warmup+args.draws)
                jax.block_until_ready(keys)
                qrows, deltas, accepts, finite = [], [], [], []
                chain_warmup = chain_sampling = 0.
                # Fixed-size timed chunks bound host overhead; timing includes result synchronization.
                for phase, n, offset in [('warmup', args.warmup, 0), ('sampling', args.draws, args.warmup)]:
                    elapsed = 0.
                    for begin in range(0, n, args.chunk):
                        pending = []
                        tick = time.perf_counter()
                        for index in range(begin, min(begin+args.chunk, n)):
                            state, out = compiled(state, keys[offset+index]); pending.append(out)
                        jax.block_until_ready((state, pending))
                        elapsed += time.perf_counter()-tick
                        if phase=='sampling':
                            for qout, delta, accepted, isfinite in pending:
                                qrows.append(np.asarray(qout)); deltas.append(float(delta))
                                accepts.append(bool(accepted)); finite.append(bool(isfinite))
                        print(f'  repeat={repeat} chain={chain} {phase} {min(begin+args.chunk,n)}/{n}', flush=True)
                    if phase=='warmup': chain_warmup=elapsed
                    else: chain_sampling=elapsed
                arrays = dict(q=np.asarray(qrows), deltaH=np.asarray(deltas), accepted=np.asarray(accepts),
                              finite=np.asarray(finite), sampling_seconds=chain_sampling, warmup_seconds=chain_warmup)
                save_npz(checkpoint, **arrays)
                allq.append(arrays['q']); alldelta.append(arrays['deltaH']); allaccepted.append(arrays['accepted']); allfinite.append(arrays['finite'])
                sampling_seconds += chain_sampling; warmup_seconds += chain_warmup
        q = np.stack(allq); finite = np.stack(allfinite)
        stats = efficiency(q, sampling_seconds, warmup_seconds, args.rhat_limit)
        # Each independent repetition also receives its own diagnostics and efficiency score.
        repeat_scores = []
        for r in range(args.repeats):
            section = q[r*args.chains:(r+1)*args.chains]
            seconds = 0.
            for c in range(args.chains):
                with np.load(folder/f'chain_{r}_{c}.npz') as saved:
                    seconds += float(saved['sampling_seconds'])
            repeat_scores.append(efficiency(section, seconds, rhat_limit=args.rhat_limit))
        diagnostics = probe(H, config, spec['method'], initial, jr.PRNGKey(args.seed+100000)) if args.probe else {}
        eligible = (stats['converged'] and finite.all() and args.draws>=500 and
                    stats['min_bulk_ess']>=args.min_ess and
                    all(s['converged'] and s['min_bulk_ess']>=args.min_ess for s in repeat_scores))
        record = dict(configuration=spec, compile_seconds=compile_seconds,
                      sampling_seconds=sampling_seconds, warmup_seconds=warmup_seconds,
                      acceptance_rate=float(np.mean(allaccepted)), nonfinite_proposals=int((~finite).sum()),
                      mean_squared_jump=float(np.mean(np.diff(q, axis=1)**2)),
                      converged=stats['converged'], eligible=bool(eligible),
                      min_bulk_ess=stats['min_bulk_ess'], seconds_per_min_bulk_ess=stats['seconds_per_min_bulk_ess'],
                      total_seconds_per_min_bulk_ess=stats['total_seconds_per_min_bulk_ess'],
                      diagnostics={**stats, 'parameters':PARAMETERS, 'per_repeat':repeat_scores,
                                   'mean_theta':np.exp(q).mean(axis=(0,1)),
                                   'sd_theta':np.exp(q).std(axis=(0,1))}, probe=diagnostics)
        atomic_json(folder/'result.json', record)
        summarize(root)
        print('DONE', folder.name, 'acceptance=', record['acceptance_rate'], 'seconds/ESS=', record['seconds_per_min_bulk_ess'], flush=True)
    print(json.dumps(serializable(summarize(root)), indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=['smoke','screen','ess'], default='screen')
    p.add_argument('--output', type=Path, default=HERE/'benchmark_results')
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--methods', nargs='+', choices=['HMC','AA'], default=['HMC','AA'])
    p.add_argument('--taus', nargs='+', type=float, default=[.005,.01,.02,.04,.08])
    p.add_argument('--times', nargs='+', type=float, default=[.5,1.])
    p.add_argument('--histories', nargs='+', type=int, default=[2])
    p.add_argument('--nodes', nargs='+', type=int, default=[2])
    p.add_argument('--tols', nargs='+', type=float, default=[1e-3])
    p.add_argument('--max-iter', type=int, default=30)
    p.add_argument('--beta', type=float, default=.95)
    p.add_argument('--ode-dt', type=float, default=.02)
    p.add_argument('--dtype', choices=['float32','float64'], default='float32')
    p.add_argument('--notebook-priors', action='store_true')
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--draws', type=int)
    p.add_argument('--warmup', type=int)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--init-scale', type=float, default=.05)
    p.add_argument('--chunk', type=int, default=25)
    p.add_argument('--rhat-limit', type=float, default=1.01)
    p.add_argument('--min-ess', type=float, default=400.)
    p.add_argument('--probe', action='store_true', help='Untimed AA residual and iteration diagnostics')
    return p


def main():
    p = parser(); a = p.parse_args()
    defaults = {'smoke':(0,8), 'screen':(10,25), 'ess':(500,1000)}
    w, d = defaults[a.preset]
    a.warmup = w if a.warmup is None else a.warmup
    a.draws = d if a.draws is None else a.draws
    if a.chains<2 or a.draws<8 or a.warmup<0 or a.repeats<1 or a.chunk<1:
        p.error('Need chains>=2, draws>=8, warmup>=0, repeats>=1, chunk>=1')
    if any(x<=0 for x in a.taus+a.times+a.tols) or any(x<1 for x in a.histories+a.nodes) or a.max_iter<1 or a.ode_dt<=0:
        p.error('Step sizes, trajectory lengths, tolerances, histories, nodes and iteration limits must be positive')
    run(a)

if __name__ == '__main__': main()
