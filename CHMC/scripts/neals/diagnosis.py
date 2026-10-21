"""
Per-(T, len) summary of the Newton chains for one τ, to locate where T = 2-5 diverge from T = 1.
Usage: python diag_newton.py [tau_idx]      (default 0, τ = 0.5)
"""
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

base = Path("/scratch/johngallagher/neals_results")
meth = "Newton"
taus = 2**-np.linspace(1, 5, 5)
Ts = np.linspace(1, 5, 5)
lens = np.logspace(2, 4, 9, base=10, dtype=int)
j = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tau = taus[j]

print(f"τ = {tau:g}.  target: yy mean 0, sd 3.  rt/step = runtime / (len · N), N = T/τ")
print(f"{'T':>3} {'len':>6} {'n_bad':>5} {'acc':>5} {'yy_mean':>8} {'yy_sd':>6} {'max|yy|':>7} "
      f"{'rt':>8} {'rt/step µs':>10} {'same_runs':>9}  written")
for T in Ts:
    N = int(round(T / tau))
    for L in lens:
        d = base / meth / f"tau_{tau:.12g}" / f"T_{T:.1f}" / f"len_{L}"
        n_bad, acc, yy_m, yy_s, yy_max, rts, mt, first = 0, [], [], [], [], [], [], []
        for r in range(10):
            p = d / f"run_{r}.npz"
            if not p.is_file():
                n_bad += 1
                continue
            with np.load(p) as f:
                q = np.asarray(f["q"])
                if q.shape[0] != L:
                    n_bad += 1
                yy = q[:, 0]
                yy_m.append(yy.mean()); yy_s.append(yy.std()); yy_max.append(np.abs(yy).max())
                rts.append(float(f["runtime"]))
                if "accepted" in f.files:
                    acc.append(np.mean(f["accepted"]))
                first.append(q[: min(5, L)].copy())
            mt.append(p.stat().st_mtime)
        if not rts:
            print(f"{T:3.0f} {L:6d}  no files"); continue
        same = sum(np.array_equal(first[0], x) for x in first[1:])      # runs identical to run 0
        rt = np.mean(rts)
        stamp = datetime.fromtimestamp(min(mt)).strftime("%m-%d %H:%M")
        print(f"{T:3.0f} {L:6d} {n_bad:5d} {np.mean(acc) if acc else np.nan:5.2f} {np.mean(yy_m):8.2f} "
              f"{np.mean(yy_s):6.2f} {np.max(yy_max):7.1f} {rt:8.2f} {1e6*rt/(L*N):10.1f} {same:9d}  {stamp}")
    print()