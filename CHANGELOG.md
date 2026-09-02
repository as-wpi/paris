# Changelog

All notable changes to PARIS are recorded here, one section per release. Versions follow
`MAJOR.MINOR.PATCH`; while the major version is 0, a minor release may add public API. Defaults
are never changed silently: a change to a default convention is a breaking change and is listed
under **Changed**.

## 0.6.0 — Unreleased

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
