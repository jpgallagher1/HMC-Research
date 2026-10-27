# October 20, 2026
# Sliced Wasserstein error for Neal's funnel (timed_blocks_funnel.py output): compute only.
# One sliced W_p value per (method, tau, T, run, chain length), computed in parallel over chains and
# saved to OUTPUT/sw_data/METHOD/tau_*/T_*/run_*.npz after every length, so an interrupted job
# resumes where it stopped. Plots are made from sw_data by timed_blocks_funnel_plots.py.
# Sliced W_p follows the p-Gauss scripts: 10_000 exact target samples, 1000 projections, p = 1, fixed seed.

import os
# One thread per worker; the parallelism is across chains.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_var] = '1'

from pathlib import Path
import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import ot

# =============================================================================
# Configuration
# =============================================================================

n_ref = 10_000              # exact funnel samples used as the reference measure
n_projections = 1000        # random directions; the same ones are used for every chain
sw_p = 1                    # sliced W_p: 1 gives sliced W_1 (mean over directions of 1D W_1)
ref_seed = 0
max_len = 100_000           # longest chain prefix evaluated; POT's peak memory is about 3 GB at 100k samples, 1000 projections

base = Path('/Users/johngallagher/Documents/UCMerced/Research/GitHub/HMC-Online/results/neals_funnel_det_samekeys_tau0.125_T3')
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=base)
parser.add_argument('--workers', type=int, default=None)    # default: SLURM_CPUS_PER_TASK, else all cores
parser.add_argument('--n_projections', type=int, default=n_projections)
parser.add_argument('--max_len', type=int, default=max_len)
parser.add_argument('--dry_run', action='store_true')       # print chain lengths and the estimated cost, compute nothing
args = parser.parse_args()
base = args.output
n_projections = args.n_projections
max_len = args.max_len
n_workers = args.workers or int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
sw_base = base / 'sw_data'

with open(base / 'metadata.json') as f:
    metadata = json.load(f)

methods = metadata['methods']
taus = metadata['taus']
Ts = metadata['Ts']
dim = metadata['dim']
block_size = metadata['block_size']

folders = list(base.glob('*/tau_*/T_*/run_*'))
if not folders:
    raise ValueError('No saved runs found.')
n_runs = max(int(folder.name.split('_')[1]) for folder in folders) + 1
nblocks = max(len(list(folder.glob('block_*.npz'))) for folder in folders)
if nblocks == 0:
    raise ValueError('No saved blocks found.')
# Chain lengths to evaluate: 200, 400, 600, 800, 1000, 2000, ..., 10_000, then log-spaced
# (15k, 20k, 30k, 50k, ...) up to max_len or what was saved. Extended chains can be much longer
# than max_len; only the first max_len samples are evaluated.
lens = np.array(sorted({n for n in [*range(200, 1000, 200), *range(1000, 10_001, 1000),
                                    *[m * 10**e for e in range(3, 9) for m in (10, 15, 20, 30, 50, 70)]]
                        if n <= min(nblocks * block_size, max_len)}))

# Exact funnel samples: y ~ N(0, 3^2), x_i | y ~ N(0, exp(y)) for the other dim - 1 coordinates.
rng = np.random.default_rng(ref_seed)
y = rng.normal(0.0, 3.0, size=n_ref)
x = rng.normal(size=(n_ref, dim - 1)) * np.exp(y / 2)[:, None]
xt = np.column_stack([y, x])                                             # (n_ref, dim)

settings = dict(n_ref=n_ref, n_projections=n_projections, p=sw_p, ref_seed=ref_seed, dim=dim)


def remaining_lengths(task, n_samples):
    """Lengths of this chain (up to n_samples) with no saved sliced W value yet."""
    meth, j, k, i = task
    wanted = lens[lens <= n_samples]
    path = sw_path(meth, j, k, i)
    if not path.exists():
        return wanted
    with np.load(path) as old:
        saved = old['lens'][np.isfinite(old['sw'])]
    return wanted[~np.isin(wanted, saved)]


def chain_folder(meth, j, k, i):
    return base / meth / f'tau_{taus[j]}' / f'T_{Ts[k]}' / f'run_{i}'


def sw_path(meth, j, k, i):
    return sw_base / meth / f'tau_{taus[j]}' / f'T_{Ts[k]}' / f'run_{i}.npz'


def compute_chain(task):
    """Sliced W_p of one chain's prefixes at every length in `lens`, saved after each length."""
    meth, j, k, i = task
    blocks, block_times = [], []
    for l in range(nblocks):
        path = chain_folder(meth, j, k, i) / f'block_{l:08d}.npz'
        if not path.exists():
            break
        with np.load(path) as result:
            blocks.append(result['q'])
            block_times.append(float(result['runtime']))
    if not blocks:
        return task, 0, 0.0
    full_chain = np.concatenate(blocks, axis=0)
    edges = np.arange(len(blocks) + 1) * block_size
    cum_time = np.concatenate([[0.0], np.cumsum(block_times)])

    out = sw_path(meth, j, k, i)
    out.parent.mkdir(parents=True, exist_ok=True)
    sw = np.full(len(lens), np.nan)
    runtime = np.full(len(lens), np.nan)
    if out.exists():                                   # reuse values already computed for these lengths
        with np.load(out) as old:
            for L, value, t in zip(old['lens'], old['sw'], old['runtime']):
                hit = np.nonzero(lens == L)[0]
                if hit.size:
                    sw[hit[0]], runtime[hit[0]] = value, t

    start = time.perf_counter()
    done = 0
    for l, length in enumerate(lens):
        if length > len(full_chain):
            break
        runtime[l] = np.interp(length, edges, cum_time)
        if np.isfinite(sw[l]):
            continue
        chain = full_chain[:length]
        if np.isfinite(chain).all():
            sw[l] = ot.sliced_wasserstein_distance(
                chain, xt, a=None, b=None, n_projections=n_projections, p=sw_p, seed=ref_seed)
        done += 1
        temporary = out.with_name(out.name + f'.tmp{os.getpid()}')
        with temporary.open('wb') as f:
            np.savez(f, lens=lens, sw=sw, runtime=runtime, n_samples=len(full_chain),
                     n_projections=n_projections, n_ref=n_ref, p=sw_p)
        os.replace(temporary, out)
    return task, done, time.perf_counter() - start


def dry_run():
    """Chain lengths per method and a rough cost estimate. Reads file names only."""
    seconds_per_1000 = 0.4 * n_projections / 1000        # measured: ~0.4 s per 1000 samples at 1000 directions
    total = total_left = 0.0
    print(f'{n_projections} directions; evaluating {len(lens)} lengths up to {max_len:,}: '
          f'{lens[:3].tolist()} ... {lens[-3:].tolist()}')
    longest = 0
    for meth in methods:
        n_samples, keys = [], []
        for j in range(len(taus)):
            for k in range(len(Ts)):
                for i in range(n_runs):
                    n = block_size * len(list(chain_folder(meth, j, k, i).glob('block_*.npz')))
                    if n:
                        n_samples.append(n)
                        keys.append((j, k, i))
        n_samples = np.array(n_samples)
        cost = sum(float(np.sum(lens[lens <= n])) for n in n_samples) / 1000 * seconds_per_1000
        left = sum(float(np.sum(remaining_lengths((meth, j, k, i), n)))
                   for (j, k, i), n in zip(keys, n_samples)) / 1000 * seconds_per_1000
        n_left = sum(remaining_lengths((meth, j, k, i), n).size > 0 for (j, k, i), n in zip(keys, n_samples))
        total += cost
        total_left += left
        longest = max(longest, min(n_samples.max(), max_len))
        print(f'  {meth:5s} {len(n_samples):4d} chains, samples min/median/max = '
              f'{n_samples.min():,} / {int(np.median(n_samples)):,} / {n_samples.max():,}; '
              f'~{cost / 3600:.1f} core-hours, {left / 3600:.1f} still to do '
              f'({n_left} chains have work left)')
    print(f'Estimated total: {total / 3600:.1f} core-hours; still to do: {total_left / 3600:.1f} core-hours, '
          f'so about {total_left / 3600 / n_workers * 60:.0f} min on {n_workers} workers '
          f'(rough; assumes perfect load balance).')
    print(f'Peak memory per worker: about {longest * n_projections * 30 / 1e9:.1f} GB '
          f'(longest prefix evaluated: {longest:,} samples).')


if __name__ == '__main__':
    if args.dry_run:
        dry_run()
        raise SystemExit
    sw_base.mkdir(parents=True, exist_ok=True)
    settings_path = sw_base / 'settings.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError('Saved sliced-Wasserstein settings differ. Use a new output directory '
                             f'or delete {sw_base}.')
    else:
        temporary = settings_path.with_name(f'settings.tmp{os.getpid()}')
        temporary.write_text(json.dumps(settings, indent=2))
        os.replace(temporary, settings_path)

    tasks = []
    for meth in methods:
        for j in range(len(taus)):
            for k in range(len(Ts)):
                for i in range(n_runs):
                    n_files = len(list(chain_folder(meth, j, k, i).glob('block_*.npz')))
                    if n_files:
                        tasks.append((n_files, (meth, j, k, i)))
    tasks = sorted(tasks, key=lambda item: -item[0])                          # longest chains first
    # Work left per chain, in samples evaluated (cost is roughly proportional to it). Chains that are
    # already complete have none, so they do not distort the time estimate when a job is resumed.
    work = {task: float(np.sum(remaining_lengths(task, block_size * n_files))) for n_files, task in tasks}
    tasks = [task for _, task in tasks]
    total_work = sum(work.values())

    print(f'{len(tasks)} chains, {len(lens)} lengths, {n_projections} projections, '
          f'{n_workers} workers; {sum(w > 0 for w in work.values())} chains have work left', flush=True)
    t0 = last_print = time.perf_counter()
    finished = computed = 0
    done_work = 0.0
    with mp.get_context('spawn').Pool(n_workers) as pool:
        for task, done, seconds in pool.imap_unordered(compute_chain, tasks, chunksize=1):
            finished += 1
            computed += done
            done_work += work[task]
            now = time.perf_counter()
            if now - last_print >= (10 if now - t0 < 300 else 60) or finished == len(tasks):
                last_print = now
                # Time left from the work finished so far; the longest chains run first, so this
                # tends to be pessimistic early on and the tail (idle workers) is not modelled.
                left = (total_work - done_work) / (done_work / (now - t0)) / 60 if done_work else float('nan')
                print(f'[{now - t0:.0f}s] {finished}/{len(tasks)} chains done, {computed} lengths computed, '
                      f'{100 * done_work / max(total_work, 1):.0f}% of the work, ~{left:.0f} min left', flush=True)
    print(f'Finished in {time.perf_counter() - t0:.0f}s; data in {sw_base}', flush=True)
