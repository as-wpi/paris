# Changelog

All notable changes to PARIS are recorded here, one section per release. Versions follow
`MAJOR.MINOR.PATCH`; while the major version is 0, a minor release may add public API. Defaults
are never changed silently: a change to a default convention is a breaking change and is listed
under **Changed**.

## 0.11.0 — 2026-09-03 — Stored fits and resumable inference for the jump models (fork: as-wpi/paris)

### Added
- `jump_fits` (and `risk_fits` / `trend_fits` for one series): the fitted model at every refit
  date as a storable DataFrame — training scaler (`mu`, `sd` per feature) and centres in
  standardised units. Pass it back as `fits=` to `jump_states` / `risk_states` / `trend_states`
  on a longer history with the same arguments: stored refits are reused, new ones fitted, and
  the result equals a cold run. `since=` restricts the online inference to observations at or
  after a date (identical values; earlier NaN) — the daily-refresh path for a screener.
  Golden cases `jump_fits/*`, `risk_fits/*`, `trend_fits/*`, `jump_states/since`; a resume test.

### Changed
- Every refit is now seeded by `(random_state, refit position)` instead of one generator
  consumed across refits, so a resumed run can reproduce a cold run exactly. Defaults and
  conventions are unchanged, but the best-of-`n_init` seeding differs per refit, and 41 of the
  90 stored golden outputs on the small test inputs (252-day windows; risk-model and derived
  tables) changed as a result. With production defaults (1,260-day windows, 10 restarts) the
  change moves 12–39 trend state-days and 0–40 risk state-days per fund on the eight benchmark
  ETFs (4,000–7,100 days each, i.e. under 1 %) — the best-of-10 tie-breaker landing on a
  different local optimum at a few refits. The research record of 2026-09-03 was produced under
  the old seeding and is not re-run; differences of this size do not touch any verdict.
- The risk model's feature is now named `vol` (was the series name) so `risk_fits` output is
  portable across series.

## 0.10.0 — 2026-09-03 — Expanding-window calibration; research sequence closes the regime-improvement thread (fork: as-wpi/paris)

### Added
- `calibration="expanding"` on `jump_states`, `jump_centers`, `risk_states`, `trend_states`,
  `joint_states`, `risk_centers` and `trend_centers`: each refit is fitted on every observation
  before it, with `window` as the minimum training length, instead of the last `window`
  observations. Default stays `"rolling"`; the online lookback is unchanged. Golden cases
  `*/expanding`; causality test.

### Research (no library change)
- `research/frozen_rule_benchmark.py` + `research/frozen_rule_benchmark/`: pre-registered benchmark of
  the untouched PARIS regimes against the rtl-screener's frozen two-term rule and S1×S2×S3 stack on
  SPY and seven 1× ETFs, plus the amended-in LLT signals (fit quality, InnovLMAR, LLT drift, trend
  strength) as second jump-model features. Verdicts: stage 1 LOSES by the letter of a mis-specified
  MaxDD clause, beats the stack 8/8 on Sharpe in substance; stage 2 four RETIREs; trend strength ≡
  LLT drift under standardisation (constant steady-state slope variance). `research/fetch_fmp.py`
  fetches the dividend-adjusted OHLC cache (git-ignored).
- `research/calibration_memory.py`: rolling 1,260-day vs expanding calibration, pre-registered;
  both RETIRE (expanding cures window anchoring but exits later in bears; trend Martin −0.17, CI
  excludes 0). `research/sizing_estimator.py`: EWMA vs Yang–Zhang, Parkinson EWMA and the
  screener's K-ATR inside the vol-target loop on 1x and 3x panels; EWMA best tracker 15/15, all
  three RETIRE. Design decision recorded: the risk model is a diagnostic, not a gate; volatility
  enters through sizing.
- Further pre-registered tests (all RETIRE; see `research/SEQUENCE_2026-09-03.md`): EWMA-weighted
  Yang–Zhang in the vol-target; the four-state momentum cycle (dynamic speeds, MED, three discrete
  rules) as a daily switch vs trend λ5; VR(5,63) and VIX term structure as switch conditions; a
  hybrid (expanding calm centre, rolling crisis centre) calibration memory.

## 0.9.0 — 2026-09-03 — Parity for the jump-model regimes: tables, signals, fitted centres, causal sizing (fork: as-wpi/paris)

### Added
- One conditional-table engine behind every state table. `state_table(returns, states, benchmark,
  rf, shift, labels)` reports count, frequency, the fund's annualised mean, volatility, population
  skewness and up-frequency, the benchmark's same four moments and the active mean when a
  benchmark is given, and the excess means when `rf` is given, for ANY label or code series. `shift=0` pairs the
  label at *T* with the return of *T* (already-lagged jump labels); `shift=1` pairs *t* with *t+1*
  (momentum states). `risk_state_table`, `trend_state_table` and `joint_state_table` now return
  these columns (plus `rf` support); `momentum_state_table` and `momentum_conditional_table`
  delegate to the same engine.
- `state_transitions(states, order)`: the transition matrix of any label or code series;
  `momentum_transitions` delegates to it.
- `risk_signal` (the annualised EWMA or rolling volatility the risk model classifies, `log=True`
  for the actual feature) and `trend_signal` (the named alias of the momentum signals).
- `jump_centers`: per refit date, the fitted centres in original feature units and, for a
  one-feature two-state model, `threshold` (the midpoint of the centres, the zero-penalty switching
  level). `risk_centers` reports them in annualised-volatility units ("risk-off above about x %");
  `trend_centers` in signal units.
- `state_sizing(returns, states, rf, window, refit, min_obs, table)`: causal state-conditional
  exposure — per-state Sharpe on the history before each refit (expanding or rolling), exposure
  `clip(SR_k / max SR, 0, 1)`, applied until the next refit; `table=True` gives the per-refit
  mapping. The library form of the walk-forward's sizing rule and the causal counterpart of
  `dynamic_speeds` for the jump-model states. Truncation tests assert causality.

### Changed
- `momentum_state_table` columns are renamed to the unified names: `mean (ann.)` → `own mean
  (ann.)`, `volatility (ann.)` → `own vol (ann.)`, `skewness` → `own skewness`, `up frequency` →
  `own up frequency`. `momentum_conditional_table` gains the vol, skewness and up-frequency
  columns. `risk_state_table` / `trend_state_table` / `joint_state_table` gain skewness and
  up-frequency and, with a benchmark, the active mean. No label or number changes.

## 0.8.0 — 2026-09-02 — Combining the risk and trend indicators (fork: as-wpi/paris)

### Added
- `combine_states(risk, trend, method)`: exposure in [0, 1] from the two binaries — `"graded"`
  (`(risk + trend) / 2`, the default), `"gate"` (trend gates, risk sizes to ½), `"and"`, `"or"`, and
  `"cells"` (an explicit exposure per joint cell, the hook for state-conditional sizing).
- `state_table(returns, states)`: count, frequency, annualised mean and volatility of the fund's
  (and benchmark's) return over the periods carrying each value of any label or code series.
- `joint_state_table`: the four cells of risk × trend with the same statistics — the evidence for
  how to combine the two indicators.
- `joint_states`: one jump model on any ordered subset of `("logvol", "slow", "fast")` with
  `n_states` states and optional `feature_weights`; states ordered by the first feature's centre;
  same rolling calibration, online inference and `lag` conventions.
- `research/walkforward_jump.py`: expanding-window, annually re-selected walk-forward of every
  indicator, combination and joint model on the US market factor and eight ETFs, 10 bp one-way
  cost; results in `research/walkforward_2026-09-02/`.

### Fixed
- `feature_weights` (in `trend_states`, now also `joint_states` and `jump_states`) were applied to
  features standardised on the **full sample** before the rolling scaler: a look-ahead through the
  full-sample mean, sd and clip, and — because the per-window scaler then divided by the weighted
  sd — no effect on the clustering at all. Weights now multiply the standardised, clipped features
  inside every training and lookback window. Values with weights change; unweighted results are
  unchanged. The truncation test now covers weighted models.

### Causality audit
- The only hindsight function is `jump_labels` (a full-sample fit, for research). No indicator,
  table or the walk-forward script calls it; every state comes from `jump_states` (rolling fit on
  the window *before* each refit date, forward-only online inference, `lag=1`). Exposure sizing and
  candidate selection in the walk-forward use history up to the prior year-end only; the script has
  no whole-history option.

### Walk-forward evidence (`research/walkforward_2026-09-02/CROSS_ASSET.md`)
- Nine series (US market factor 1995–2026; SPY, QQQ, XLK, IWM, EFA, EEM, TLT, GLD from 2002–2013
  to 2026), t+0 execution, 10 bp one-way, T-bill cash. On the eight ETFs — untouched by any tuning —
  trend-on/off (λ = 5) is the best single rule: Sharpe above buy-and-hold on 5 of 8 (mean 0.51 vs
  0.46), max drawdown better on 8 of 8 (mean −25 % vs −45 %), Martin on 7 of 8 (1.12 vs 0.75).
  The combinations buy drawdown protection rather than Sharpe: `gate` −22 % mean max drawdown at
  buy-and-hold Sharpe, `graded` similar; `and` too restrictive, `or` gives the protection back.
  Risk-on/off alone is weak outside US equities. Annual re-selection over ~100 candidates is
  harmful (2 of 8), and the joint multi-feature models with causal sizing do not beat the two
  binaries. Every rule gives up CAGR: these are drawdown gates, not return signals. No default
  changed on this evidence; `combine_states` keeps `"graded"` as its default.

## 0.7.0 — 2026-09-02 — Jump-model risk-on/off and trend-on/off indicators (fork: as-wpi/paris)

### Added
- `jump`: a pure numpy statistical jump model (Nystrup, Lindström & Madsen 2020) with the loss and
  penalty conventions of the reference `jumpmodels` package, against which the test suite checks
  fit and online inference exactly (`uv sync --group oracle`). `jump_labels` is the hindsight
  path; `jump_states` is the causal indicator: rolling calibration (`window` observations before
  each refit date, `refit="ME"` by default), online forward-DP inference over a `lookback`
  window, training-window standardisation and clipping.
- `risk_states` / `risk_state_table`: risk-on (1) / risk-off (0) from a one-feature jump model on
  log volatility — the RiskMetrics EWMA (λ = 0.94, 60-observation warm-up) of daily log returns by
  default, or a rolling sample sd — on the series' own returns (`basis="own"`) or the benchmark's
  (`basis="benchmark"`, bundled S&P 500 proxy by default). Low volatility is risk-on. The value for
  day *T* is the online state at *T-1* (`lag=1`). The table reports count, frequency and the
  annualised arithmetic mean and volatility of the labelled periods, for the fund and — with the
  benchmark basis — for the benchmark.
- `trend_states` / `trend_state_table`: trend-on (1) / trend-off (0) from a jump model on the
  momentum turning-point signals (`("slow", "fast")` by default, `("slow",)` for the slow signal
  alone; `feature_weights`), on a raw, excess or relative basis; the high slow-signal state is
  trend-on. Same calibration, inference and lag conventions.
- `momentum_conditional_table`: by momentum state at *t*, the annualised arithmetic mean of the
  return at *t+1* of the fund, the benchmark and the active return (and the excess versions with
  `rf`) side by side.
- `regime_runs`: contiguous runs (`start`, `end`, `state`, `length`) of any label series — the
  input to a colour-coded regime ribbon; README shows the matplotlib recipe.

### Conventions
- Every jump-model indicator is causal: nothing after *T-1* enters the value reported for *T*; the
  test suite asserts that truncating the history leaves earlier labels unchanged. Warm-up
  (feature warm-up + `window` + `lag`) is NaN.
- Default penalties were sized on the daily US market factor (Fama–French, 1992–2026; 1,260-day
  window, monthly refit, `lag=1`, 0/1 strategy in cash when off, evaluated in sample over a
  five-value grid — a sizing, not a validation). Risk (`RISK_PENALTY = 50`): the EWMA log-vol
  feature is smooth, so switches are ~1/yr even at λ = 5; from λ = 50 the low-vol state carries the
  higher arithmetic mean (12.9 % vs 11.2 % ann.), the 0/1 strategy's Sharpe is 0.59 vs 0.53
  buy-and-hold and its max drawdown −21 % vs −55 %. Below 50 the high-vol state has the *higher*
  mean (V-shaped recoveries) and the 0/1 Sharpe is below buy-and-hold, though drawdowns still
  halve — a risk gate, not an alpha signal. Trend (`TREND_PENALTY = 5`): the slow/fast features
  are persistent themselves, so the lightest penalty on the grid wins — 0/1 Sharpe 0.75 vs 0.60 for
  the raw 252-day sign on the same series and 0.52 buy-and-hold, max drawdown −20 % vs −28 %, 1.9 vs
  2.6 switches/yr, on-state mean 15.5 % vs 5.3 % off; the slow signal alone reaches 0.69. Both
  defaults are exposed as `jump_penalty`; re-size them for other assets or frequencies.

## 0.6.0 — 2026-09-02 — Regimes, drawdown distributions, Kelly and deflated Sharpe (fork: as-wpi/paris)

### Added
- `regimes`: market regimes from slow and fast time-series momentum after Goulding, Harvey &
  Mazzoleni (2023, *Journal of Financial Economics* 149(3), 378–406). `momentum_states` labels
  every period Bull / Correction / Bear / Rebound from the signs of the trailing slow and fast
  arithmetic-mean returns (12/1 monthly, 252/21 daily, 52/4 weekly, 4/1 quarterly by default;
  `compound=True` for compounded trailing returns; `codes=True` for integers). `basis` chooses the
  return series the signals see: `"raw"`, `"excess"` (minus `rf`, the paper's construction) or
  `"relative"` (minus `benchmark`, the bundled S&P 500 proxy by default). Also
  `momentum_signal`, `momentum_state` (latest), `momentum_state_age`, `momentum_state_table`
  (the paper's Figure 1: subsequent-return moments by state), `momentum_transitions` (Table 7),
  `momentum_speed_weights` (equation 11 / 30 positions for a static or state-dependent speed) and
  `dynamic_speeds` (Proposition 9 closed form, clipped to [0, 1]). Reproduces the paper's US
  frequencies, conditional moments, transition matrix and DYN speeds from the Fama–French market
  factor (1969–2018) and matches a daily 252/21 close-based classifier exactly.
- `Portfolio` exposes the regime functions with `rf` and `benchmark` pre-filled.
- `rolling_ulcer(returns, window)`: vectorised Ulcer Index over a sliding window, drawdowns
  measured from the running peak of the whole history (`trim`, `ddof`, `pct` as elsewhere).
- `drawdown_distribution`: distribution table of the per-period drawdowns or of the rolling Ulcer
  Index (`stat="ulcer"`) — share below a threshold, mean, quantiles of the magnitude, maximum.
- `ulcer_index(pct=True)` reports percent, as in Martin's original.
- `calmar_ratio` / `sterling_ratio` take `window` (trailing observations, e.g. Young's 36 months).
- `calmar_ratio`, `sterling_ratio`, `martin_ratio`, `pain_ratio`, `burke_ratio` take
  `method="calendar"` (`days_in_year`, `start`) to annualise over elapsed calendar days, the
  convention of `cagr(method="calendar")`. Defaults are unchanged.
- `kelly_ratio` takes `fraction` (fractional Kelly; `half=True` is `fraction=0.5`) and
  `excess_var=True` (variance of the excess rather than the raw return). The docstring names the
  convention: continuous-time Kelly leverage ``mean(r - rf) / var(r)`` (Merton 1969; Thorp 2006).
- `kelly_interval`: delta-method confidence interval for the Kelly leverage (`lower`, `kelly`,
  `upper`), sampling error of the mean and of the variance under normality.
- `deflated_sharpe` (Bailey & López de Prado 2014): the probabilistic Sharpe ratio against the
  expected maximum Sharpe of `trials` independent strategies (a count with `sharpe_variance`, or
  the sequence of the trials' per-period Sharpe ratios); equals `probabilistic_sharpe` for one trial.
- `min_track_record` (Bailey & López de Prado 2012): observations (or `years=True`) needed for the
  probabilistic Sharpe ratio to reach `confidence` against `benchmark_sharpe`; `+inf` when the
  Sharpe does not exceed the benchmark.

### Changed
- The `stats()` row `Kelly (half)` is now labelled `Kelly leverage (half)` — the number is a
  multiple of wealth, not a ratio. Selecting it by name (`metrics=[...]`) must use the new label.

## 0.5.1 — 2026-08-28 — Initial public release

### Added
- 108 scalar statistics across `returns`, `risk`, `drawdown`, `ratios` and `relative` — each a
  pure function, Series → scalar, DataFrame → Series by column — plus `drawdown_table`,
  `calendar_returns`, `period_returns`, `regression_stats` and the `stats()` summary table with
  its `ABSOLUTE_METRICS` / `RELATIVE_METRICS` registries.
- `tables`: capture, downside, distribution, annualised-returns, calendar (month grid + annual),
  drawdown-summary and drawdown-ratio tables, and `rolling(fn, window)` over any scalar function;
  every cell is a call to the corresponding scalar function.
- `attribution`: `portfolio_return` from a weight vector or dated weight table with buy-and-hold
  drift and optional rebalancing, `contribution`, `bop_weights` / `eop_weights`,
  `period_contributions`, `active_contribution`, and `brinson` (Brinson–Fachler or
  Brinson–Hood–Beebower; Carino, Menchero or arithmetic linking).
- `budgeting`: Euler contributions to volatility, VaR (Gaussian, Cornish-Fisher) and CVaR
  (historical, Gaussian, Cornish-Fisher) for one weight vector; `marginal_var` / `marginal_cvar`.
- `Portfolio`: a convenience wrapper that pre-fills `benchmark`, `rf`, `periods_per_year` and
  optional weights for every function and table.
- `paris.data`: two frozen sample datasets (`load_managers`, `load_prices`, `describe`) used by
  every example and by the test suite.
- Optional extras: `paris[scipy]` (p-values in `regression_stats`), `paris[polars]` (polars
  frames accepted as input).
- Golden-value test suite (`tests/`, 731 tests): every public function on the sample data against
  its frozen result, plus shape, error-path and degenerate-input checks.

### Conventions
- Inputs are trimmed to their common window; any interior gap raises `GapError` — nothing is ever
  filled, interpolated or dropped. Frequency is inferred from the index (override with
  `periods_per_year`); insufficient samples give `NaN`, bad inputs raise a `ParisError` subclass.
- A scalar `rf` is an **annual** rate, de-annualised geometrically; a Series `rf` is per-period.
  `mar` is per-period. Drawdowns are negative; VaR / CVaR are returned as (negative) returns.
- A weight row dated *d* applies to returns strictly after *d*; rows must sum to 1 within 1e-4
  (`normalize=True` rescales); risk contributions take one weight vector.
- Defaults follow the industry-standard R reference package; every convention that differs
  between the reference implementations and the public portfolio tools is a named keyword switch
  (`ddof`, `method`, `annualize`, `geometric`, `compounding`, ...), documented in the docstring.

### Validation
- The development suite behind this release holds 1,016 automated tests (24 documented skips):
  2,080 values reproduced from the R reference package at rtol 1e-7 plus 20 drawdown tables and the
  summary tables; 49 functions of the Python reference package at rtol 1e-6; two public web tools
  at their display precision plus an independent-data check; hypothesis-based invariants;
  99–100 % line and branch coverage on the metric modules.

### Requirements
- Python >= 3.11 (verified on 3.11, 3.12 and 3.13); numpy >= 1.23; pandas >= 2.2.
