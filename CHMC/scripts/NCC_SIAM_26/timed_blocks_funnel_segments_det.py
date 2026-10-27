# October 26, 2026
# Diagnostic for the sliced W jump at long chain lengths: split each saved chain into consecutive
# segments and report, per segment, the range of y, the largest |x|, the acceptance rate and the
# sliced W_1 of the segment alone against the exact reference. Reads only block files; writes
# OUTPUT/segment_diagnostics.csv and prints a table per (method, tau, T).

import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_var] = '1'

from pathlib import Path
import argparse
import csv
import json

import numpy as np
import ot

base = Path('/Users/johngallagher/Documents/UCMerced/Research/GitHub/HMC-Online/results/neals_funnel_det_samekeys_tau0.125_T3')
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=base)
parser.add_argument('--segment', type=int, default=5000)        # samples per segment
parser.add_argument('--n_projections', type=int, default=200)
parser.add_argument('--n_ref', type=int, default=10_000)
parser.add_argument('--ref_seed', type=int, default=0)
args = parser.parse_args()
base = args.output

with open(base / 'metadata.json') as f:
    metadata = json.load(f)
methods, taus, Ts, dim = metadata['methods'], metadata['taus'], metadata['Ts'], metadata['dim']

# Same exact reference as timed_blocks_funnel_sw_det.py.
rng = np.random.default_rng(args.ref_seed)
y = rng.normal(0.0, 3.0, size=args.n_ref)
xt = np.column_stack([y, rng.normal(size=(args.n_ref, dim - 1)) * np.exp(y / 2)[:, None]])

rows = []
for meth in methods:
    for tau in taus:
        for T in Ts:
            folders = sorted((base / meth / f'tau_{tau}' / f'T_{T}').glob('run_*'),
                             key=lambda p: int(p.name.split('_')[1]))
            if not folders:
                continue
            print(f'\n{meth}  tau={tau}  T={T}')
            print(f'{"run":>4} {"samples":>8} {"segment":>13} {"y min":>7} {"y max":>7} '
                  f'{"max|x|":>8} {"accept":>7} {"SW1":>7}')
            for folder in folders:
                run = int(folder.name.split('_')[1])
                blocks = [np.load(p) for p in sorted(folder.glob('block_*.npz'))]
                if not blocks:
                    continue
                q = np.concatenate([b['q'] for b in blocks])
                acc = np.concatenate([b['accepted'] for b in blocks]).astype(float)
                for a in range(0, len(q), args.segment):
                    b = min(a + args.segment, len(q))
                    if b - a < args.segment // 2:        # skip a short tail; its SW is not comparable
                        continue
                    s = q[a:b]
                    sw = ot.sliced_wasserstein_distance(s, xt, n_projections=args.n_projections,
                                                        p=1, seed=args.ref_seed)
                    row = dict(method=meth, tau=tau, T=T, run=run, samples=len(q), start=a, stop=b,
                               y_min=s[:, 0].min(), y_max=s[:, 0].max(),
                               max_abs_x=np.abs(s[:, 1:]).max(), accept=acc[a:b].mean(), sw1=sw)
                    rows.append(row)
                    print(f'{run:>4} {len(q):>8} {f"{a}-{b}":>13} {row["y_min"]:>7.2f} {row["y_max"]:>7.2f} '
                          f'{row["max_abs_x"]:>8.1f} {row["accept"]:>7.2f} {sw:>7.3f}')

out = base / 'segment_diagnostics.csv'
with out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f'\nSaved {out}')
