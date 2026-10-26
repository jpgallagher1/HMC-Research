Edit the **Configuration** section at the top of `timed_blocks.py`, then run
from the repository root in `HMC-Research`:

```sh
python CHMC/scripts/NCC_SIAM_26/timed_blocks.py
python CHMC/scripts/NCC_SIAM_26/timed_blocks_plots.py
```

The sampling script follows the existing sweep/notebook layout: configuration,
derived quantities, then explicit method/tau/T/run/block loops. The main
computation is `sample_block`: a JIT-compiled `jax.lax.scan` of 200 transitions
returning the final carry and all 200 states. `time_block` measures that call
outside JIT and returns `(runtime, last, states)`. Plotting is in
a separate script, with arrays indexed as `(tau, T, length, run)` as in the
p-chi notebook.

Defaults are 200 consecutive transitions per block, 10,000 per chain, 10 runs,
methods LF/AA/N_K/FPI, taus `[0.0625, 0.1, 0.125]`, and Ts `[1, 3, 5]`.
The precision is `[[1, rho], [rho, 1]]`, where `rho=511/513` and kappa=512.
The reference covariance is its inverse; momentum mass is identity.
`max_iter` sets the Newton cap K and the AA/FPI cap; `m` is AA history size.
The integrators and stopping criteria are the ones used in the supplied
p-chi samplers. `gen_configs` rounds N=T/tau; actual T=N*tau is saved per block.

Each `METHOD/tau_*/T_*/run_*/block_*.npz` saves only the new block:
`q`, `p`, `deltaHs`, `accepted`, `runtime`, `start`, `stop`,
`start_qp`, `end_qp`, `end_deltaH`, `end_accepted`, `next_key`, `tau`, `T`, `N`.
A temporary file is replaced after a block finishes saving. Re-running restores
the last block's state and key; earlier blocks are preserved. Increase samples
or runs to extend a sweep. Use one writer per output directory. Settings/grid
changes require a fresh directory. Earlier schema-1 results from the initial
runner also require a separate directory; they are not overwritten.

Compilation and a disposable warm-up block are excluded from timing, as are key
generation and saving. Each timed block waits for all JAX outputs. These are
continuous chains with no statistical burn-in discarded.

The plotting script computes covariance errors using every retained position,
including repetitions after rejection. It saves `cov_metrics.npz` and p-chi-style
plots in `OUTPUT/plots`: faint individual runs, bold means, method markers,
serif text, semilog/log-log axes, iterations and cumulative time. Both maximum
diagonal error and maximum entry error are plotted. Missing prefixes are NaN;
means use the available runs at each prefix. Cumulative time is computed here
by summing saved block durations; the sampling code saves only block time.

Optional overrides support short runs without editing the configuration:

```sh
python CHMC/scripts/NCC_SIAM_26/timed_blocks.py --output /tmp/cov_smoke --samples 400 --runs 1 --taus 0.05 --Ts 0.1
python CHMC/scripts/NCC_SIAM_26/timed_blocks_plots.py --output /tmp/cov_smoke
python -m unittest discover -s CHMC/scripts/cov/tests -v
```

Smoke tests check restarts and saved covariance metrics. They do not establish
statistical convergence or a method's efficiency advantage.
