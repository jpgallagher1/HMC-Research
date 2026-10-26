# Lotka–Volterra AA versus HMC

Run from the repository root in **HMC-Research**. This suite measures actual
wall-clock seconds per minimum rank-normalized bulk ESS across all seven
parameters. It also reports tail ESS, folded rank-normalized split R-hat,
physical-parameter means/standard deviations, acceptance and nonfinite proposals.

```sh
conda run -n HMC-Research python -m pytest CHMC/scripts/lotka_volterra/benchmark_tests -q
conda run -n HMC-Research python CHMC/scripts/lotka_volterra/benchmark_ess.py \
  --preset smoke --chains 2 --output /tmp/lv_ess_smoke
```

The smoke run has only one Hamiltonian step per proposal and eight draws per
chain; it verifies the pipeline, **not** an efficiency advantage.

## Stage 1: inexpensive screening

Both methods get the same step-size/trajectory-length grid. By default there
are 10 HMC and 10 AA configurations; AA starts at history 2, two quadrature nodes,
and tolerance 1e-3. The model uses float32 and fixed ODE steps of 0.02.

```sh
conda run -n HMC-Research python CHMC/scripts/lotka_volterra/benchmark_ess.py \
  --preset screen --taus .005 .01 .02 .04 .08 --times .5 1 \
  --chains 2 --probe --output /tmp/lv_ess_screen
```

Screening uses 10 warmup and 25 sampling transitions per chain. It measures
runtime, acceptance and failures; its ESS estimates cannot establish a winner.
Inspect `summary.csv`. The optional untimed AA probe reports actual fixed-point
residuals, the solver's reported errors, iteration counts, cap hits and nominal
gradient evaluations on one seeded trajectory. The probe is diagnostic and is
not included in sampling time or an ESS denominator.

For shortlisted AA step sizes and lengths, screen solver choices separately:

```sh
conda run -n HMC-Research python CHMC/scripts/lotka_volterra/benchmark_ess.py \
  --preset screen --methods AA --taus .02 .04 --times .5 \
  --histories 2 4 --nodes 2 4 --tols .001 .01 --probe \
  --chains 2 --output /tmp/lv_ess_solver_screen
```

The above values are a proposed search grid, not demonstrated optima. Extend the
HMC search if its best result lies at an edge; tune both methods independently.
Shortlisting should consider mixing, not acceptance alone.

## Stage 2: measure ESS for finalists

Run each selected HMC/AA configuration, with at least four chains and enough
warmup to reach the posterior. All options below are examples, not selected
winners. Use different output directories for different draw counts or models.

```sh
conda run -n HMC-Research python CHMC/scripts/lotka_volterra/benchmark_ess.py \
  --preset ess --methods HMC AA --taus .01 .02 .04 --times .5 1 \
  --chains 4 --warmup 500 --draws 1000 --repeats 2 \
  --output /tmp/lv_ess_finalists
```

Each repeat uses new independent seeds, matched between methods. `--repeats 2`
means two sets of four chains, not two timing repetitions of the same draws.
Each repeat also has its own ESS/R-hat diagnostics. Increase draw count and
warmup if convergence or ESS requirements fail. No thinning or filtering of
rejected states is performed.

`comparison.json` selects the best *eligible* configuration of each method and
reports `AA_speedup_over_HMC = HMC seconds/ESS / AA seconds/ESS`. Values above 1
favor AA. It also reports paired-repeat speedups and, for eligible finalists,
physical-parameter mean differences scaled by their combined estimated Monte
Carlo standard errors. Large differences need investigation before interpreting
a speedup. Eligibility requires all finite proposals, at least 500 draws per
chain, finite diagnostics, R-hat <= 1.01, and minimum bulk ESS >= 400, both pooled
and separately in each repeat. Change thresholds with `--min-ess` and
`--rhat-limit`. Absence of eligible candidates is reported without declaring a
winner. Inspect tail ESS and physical-parameter summaries before interpreting a
speedup. Repeat the finalists on fresh seeds after tuning to limit selection
bias. These checks do not establish target invariance of the AVF proposal.

## Model and timing conventions

* `lv_model.py` uses the notebook's observations, likelihood and log-coordinate
  parameterization. It **corrects the prior indexing mistakes by default**.
  Use `--notebook-priors` to reproduce the original erroneous priors explicitly.
  The notebook itself is not changed by running the benchmark.
* ODE settings, prior choice, dtype and data checksum are recorded in
  `manifest.json` and held constant across candidates. Changing ODE step size
  changes the numerically approximated target; use a separate experiment and
  verify posterior/gradient agreement under refinement before claiming gains.
* Both methods use identity mass and the repository's existing integrators.
  The benchmark wrapper splits independent momentum and acceptance keys,
  explicitly rejects nonfinite proposals and preserves rejected chain states.
  It does not secretly substitute another nonlinear solver or rejection rule
  for finite but unconverged AA outputs.
* Warmup is discarded burn-in with fixed settings, not mass/step-size adaptation.
  Initial states are independent perturbations around the prior center. For
  difficult posteriors this may be insufficient; R-hat/ESS gate the result.
* Compilation and one execution warm-up are excluded from sampling time.
  Compilation time, warmup time, steady-state seconds/ESS and warmup-inclusive
  seconds/ESS are reported separately. Compilation is included in neither ESS
  score. Chains run serially on the selected JAX device.
* A compiled single transition is dispatched in fixed-size chunks, with
  `block_until_ready` before stopping each timer. Timings include transition
  dispatch and synchronization, but exclude host output conversion, disk I/O,
  ESS calculations and optional solver probes. This avoids the expensive
  enclosing nested-scan compilation and shows progress every chunk.
* Completed chains and configuration summaries are published atomically. Rerun
  the identical command to skip completed work. Changes to the model, source,
  hardware, seed or experiment settings require a fresh output directory.
  A crash inside a chain restarts that chain; completed chains are retained.
* Gradient-count estimates are nominal: JAX optimization, quadrature batching
  and ODE differentiation affect actual cost. Wall time is the main score.

## Automated validation

The focused tests check ESS against independent and autocorrelated sequences,
reject stuck and shifted-chain winners, detect scale disagreement with folded
R-hat, check AA against the analytic quadratic AVF step in float32/float64,
verify independent random keys and nonfinite rejection, enforce actual
`T=N*tau`, exercise artifact/resume behavior on a Gaussian target, compare the
Lotka–Volterra gradient with finite differences and check ODE refinement at the
initial center. This is separate from the repository's older incompatible
QP-constructor tests.

Dependencies: the existing HMC-Research environment needs JAX, JAXopt, Diffrax,
NumPy, SciPy, NumPyro and pytest. No ArviZ installation is required.
