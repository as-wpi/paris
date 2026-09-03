# PARIS — Portfolio Analytics, Risk & Investment Statistics

Performance statistics for a fund or a collection of funds, on an absolute basis and relative to a
benchmark. Pure Python (numpy + pandas), one small module per topic, and every number cross-checked
against independent reference implementations and public portfolio tools.

## Installation

Installs and runs on Linux, macOS and Windows with Python >= 3.11.

```bash
pip install paris-analytics             # or: uv add paris-analytics   (the import name is `paris`)
pip install "paris-analytics[scipy]"    # optional extra: p-values in regression_stats
pip install "paris-analytics[polars]"   # optional extra: accept polars frames as input
```

## Quick start

The package ships sample data (see [Sample data](#sample-data-parisdata) below), so every example
runs out of the box — `import paris` also exposes `paris.data`:

```python
import paris

m = paris.data.load_managers()              # monthly total returns, 2010-2025: six funds, three benchmarks, T-bill
funds, spx, rf = m.iloc[:, :6], m["SPY"], m["TBILL3M"]

paris.sharpe(funds, rf=rf)                  # one value per fund (Series); a Series input returns a float
paris.max_drawdown(funds)
paris.drawdown_table(m["FCNTX"], top=5)
paris.beta(funds, spx)                      # benchmark-relative: pass the benchmark second
paris.information_ratio(funds, spx)
paris.up_capture(funds, spx)
paris.period_returns(funds)                 # MTD, QTD, YTD, 1Y, 3Y, 5Y, 10Y, ITD
paris.stats(funds, benchmark=spx, rf=rf)    # the whole table, metrics x funds (benchmark last)

paris.calendar_table(m["FCNTX"], spx)       # years x Jan..Dec, Annual, SPY  (tables: rows = metrics/periods, cols = funds)
paris.downside_table(funds, spx, rf=rf)     # also capture_, distribution_, annualized_, drawdown_ratio_table, drawdown_summary
paris.rolling(funds, paris.sharpe, 36, rf=rf)   # any scalar function over a 36-observation window, date x fund

pf = paris.Portfolio(funds, benchmark=spx, rf=rf)
pf.sortino(), pf.alpha(), pf.stats(), pf.calendar_table(), pf.rolling(paris.beta, 36)

w = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]                # weights: one-time vector (or a DataFrame of dated rows)
paris.portfolio_return(funds, w, rebalance="QE")  # buy-and-hold drift between quarterly rebalances
paris.contribution(funds, w)                      # BOP weight x return per asset; period_contributions() links spans
sleeves = funds[["FCNTX", "DODGX"]].set_axis(["Growth", "Value"], axis=1)   # Brinson categories match by column name
style   = m[["IWF", "IWD"]].set_axis(["Growth", "Value"], axis=1)
paris.brinson(sleeves, [0.6, 0.4], style, [0.5, 0.5])     # allocation / selection / interaction per category
paris.volatility_contribution(funds, w, pct=True) # Euler risk shares; also var_/cvar_contribution, marginal_var
paris.Portfolio(funds, weights=w, benchmark=spx, rf=rf).sharpe()        # every stat on the weighted portfolio

paris.momentum_states(funds)                      # Bull / Correction / Bear / Rebound per month (Goulding-Harvey-Mazzoleni 2023)
paris.momentum_state(funds, basis="excess", rf=rf)          # latest state on excess returns (the paper's construction)
paris.momentum_state(funds, basis="relative")               # ... relative to the bundled S&P 500 proxy (or benchmark=)
paris.momentum_state_table(spx), paris.momentum_transitions(spx)  # subsequent-return moments by state; transition matrix
paris.momentum_speed_weights(spx, speeds=paris.dynamic_speeds(spx.loc[:"2017"]))  # state-dependent momentum positions
paris.momentum_conditional_table(funds, spx, rf=rf)  # by state: next-period mean of own, benchmark, active and excess returns

d = paris.data.load_prices().pct_change().dropna()  # daily returns
paris.risk_states(d["SPY"], window=252)             # 1 risk-on / 0 risk-off: rolling jump model on EWMA log-vol, known at T-1
paris.risk_state_table(d["FCNTX"], basis="benchmark", window=252)   # mean/vol by market risk state, fund and S&P 500
paris.trend_states(d, window=252)                   # 1 trend-on / 0 trend-off: jump model on the slow/fast momentum signals
r, t = paris.risk_states(d["SPY"], window=252), paris.trend_states(d["SPY"], window=252)
paris.combine_states(r, t, "graded")                # exposure 0 / ½ / 1 from the two binaries (also gate / and / or / cells)
paris.joint_state_table(d["SPY"], risk_kwargs={"window": 252}, trend_kwargs={"window": 252})  # mean/vol in the four cells
paris.joint_states(d["SPY"], features=("logvol", "slow", "fast"), n_states=4, window=252)      # one model, several features
paris.state_table(d["SPY"], r, benchmark=d["FCNTX"], rf=0.02)  # the unified conditional table for any label series
paris.state_transitions(r), paris.risk_signal(d["SPY"]), paris.risk_centers(d["SPY"], window=252)  # transitions, feature, fitted threshold
paris.trend_states(d["SPY"], window=252, calibration="expanding")  # centres from all history before each refit (window = minimum)
paris.state_sizing(d["SPY"], r * 2 + t, refit="QE")   # causal per-state exposure (Sharpe-scaled, re-estimated each refit)
paris.regime_runs(paris.momentum_states(d["SPY"]))  # contiguous spans (start, end, state, length) for a colour-coded ribbon
```

A colour-coded regime ribbon from `regime_runs` (matplotlib; one `axvspan` per run):

```python
import matplotlib.pyplot as plt
colors = {"Bull": "#16a34a", "Correction": "#d97706", "Bear": "#dc2626", "Rebound": "#0072B2"}
runs = paris.regime_runs(paris.momentum_states(d["SPY"]))
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(paris.wealth_index(d["SPY"]), color="black", lw=1)
for _, r in runs.iterrows():
    ax.axvspan(r["start"], r["end"], color=colors[r["state"]], alpha=0.25, lw=0)
```

```python
```

Inputs: pandas Series/DataFrame with a DatetimeIndex (daily, weekly, monthly, quarterly, yearly —
frequency is inferred, override with `periods_per_year`), numpy arrays, or polars frames (with the
`polars` extra). Series are trimmed to their common window; interior gaps raise `GapError`
(nothing is ever filled). A scalar `rf` is an **annual** rate; a Series `rf` (like `TBILL3M`
above) is per-period. Drawdowns are negative; VaR/CVaR are returned as (negative) returns.

Every function's docstring states its formula and the convention behind each keyword switch
(`help(paris.sharpe)`). A guided tour of the most used statistics, one cell each, is in
[`notebooks/paris_tour.ipynb`](https://github.com/ekam-a3/paris/blob/main/notebooks/paris_tour.ipynb)
(committed with its outputs, so it reads without running; to run it, open it in any Jupyter whose
kernel has `paris` installed).

## Sample data (`paris.data`)

PARIS ships two small frozen datasets that every example uses:

| Loader | Contents | Window |
|---|---|---|
| `paris.data.load_managers()` | monthly simple **total** returns of six widely held active US large-cap funds (`FCNTX` Fidelity Contrafund, `AGTHX` Growth Fund of America, `FMAGX` Fidelity Magellan, `AMCPX` AMCAP, `DODGX` Dodge & Cox Stock, `PRGFX` T. Rowe Price Growth Stock), three total-return benchmark proxies (`SPY`, `IWF`, `IWD` — ETFs, so net of their fees) and `TBILL3M`, the 3-month Treasury bill yield per month for use as a per-period `rf` | 192 month-ends, 2010-01-31 – 2025-12-31 |
| `paris.data.load_prices()` | daily total-return index levels of `SPY` and `FCNTX` (`.pct_change()` for returns) | 1,255 trading days, 2021-01-04 – 2025-12-31 |

`paris.data.describe()` lists every column with its full name and role. Total returns reinvest each
distribution on its ex-date; the frames are rectangular (no missing values anywhere), so they pass
`GapError` checks whole. The data is illustrative — it plays no part in the validation evidence. The
frozen CSVs ship inside the package (`src/paris/data/managers.csv`, `prices.csv`); nothing
vendor-specific ships.

## Modules (copy one file at a time — each depends only on `_core.py`)

| Module | Contents |
|---|---|
| `returns.py` | total / annualised return, CAGR, cumulative & wealth index, period returns, calendar tables, best/worst, win rate, streaks, excess & active returns |
| `risk.py` | volatility, downside/upside/semi deviation, skewness, kurtosis, VaR & CVaR (historical, gaussian, Cornish-Fisher), tail & outlier ratios |
| `drawdown.py` | drawdown series & episode table, max/average/longest drawdown, ulcer & pain index, rolling ulcer, drawdown / rolling-ulcer distribution table, Calmar, Sterling (with trailing `window`), Burke, Pain, Martin ratios (period-count or calendar-day annualisation), recovery factor |
| `ratios.py` | Sharpe (incl. VaR/ES-based, smart, probabilistic, deflated, adjusted), minimum track record, Sortino, Omega, Kappa, upside potential, profit factor, gain-to-pain, payoff, CPC, common-sense, Kelly leverage (fractional, with a confidence interval), risk of ruin, prospect, serenity |
| `relative.py` | beta/alpha (CAPM, bull/bear, timing), Jensen, Treynor, tracking error, information ratio, active premium, systematic/specific/total risk, appraisal, Fama beta, selectivity, M², capture & number/percentage ratios, batting average, regression table |
| `tables.py` | capture, downside, distribution, annualised-returns, calendar (month grid + annual), drawdown summary and ratio tables, `rolling(fn, window)` — every cell is a call to a topic-module function (imports the topic modules) |
| `attribution.py` | weights → portfolio return with drift / rebalancing, per-period contributions, BOP/EOP weights, linked multi-period contributions, active contribution, Brinson attribution (BF/BHB; Carino, Menchero or arithmetic linking) |
| `budgeting.py` | Euler contributions to volatility, VaR (Gaussian, Cornish-Fisher) and CVaR (historical, Gaussian, Cornish-Fisher) for one weight vector; marginal VaR / CVaR |
| `regimes.py` | Goulding–Harvey–Mazzoleni (2023) momentum turning points: slow/fast signals, Bull / Correction / Bear / Rebound state series on a raw, excess-of-rf or benchmark-relative basis (S&P 500 default), latest state and its age, conditional subsequent-return tables (own / benchmark / active, excess), transition matrix, regime run table, intermediate-speed strategy weights and the Sharpe-maximising dynamic speeds (imports `paris.data` lazily for the default benchmark) |
| `jump.py` | statistical jump model (pure numpy, checked exactly against `jumpmodels`): hindsight labels, causal rolling- or expanding-calibration / online-inference states, the two binary indicators — risk-on/off on log volatility (own or S&P 500 basis) and trend-on/off on the slow/fast momentum signals — their combinations (`combine_states`), joint multi-feature models (`joint_states`) and conditional return tables by state or joint cell (imports `regimes`) |
| `summary.py` / `portfolio.py` | `stats()` table and the `Portfolio` convenience wrapper (these two import all topic modules) |

Defaults follow the industry-standard R reference package; every convention that differs between
the reference implementations and the public portfolio tools is a named keyword switch (`ddof`,
`method`, `annualize`, `geometric`, `compounding`, ...) rather than a silent choice.

## Validation

The development suite behind each release holds 1,016 automated tests (24 documented skips):
2,080 values reproduced from the R reference package at rtol 1e-7 plus 20 drawdown tables and the
full set of summary tables; 49 functions of the Python reference package at rtol 1e-6; two public
web tools at their display precision plus an independent-data check; hypothesis-based invariants
(drawdown bounds, return identities, scale invariance, tail-risk ordering, regression recovery);
99–100 % line and branch coverage on the metric modules. Where a reference disagrees with PARIS
its source was read and the case excluded by name with a written reason — tolerances are never
widened to make a test pass.

## Testing

```bash
uv sync                       # or: pip install pytest   (add --all-extras for scipy p-values)
uv run pytest                 # ~15 s, no network
```

Every public function is run on the shipped sample data and compared with its frozen result in
[`tests/expected/`](https://github.com/ekam-a3/paris/tree/main/tests/expected) (one JSON file per
module, one value per line); the suite also checks input/output shapes (Series in, one number out;
DataFrame in, one value per column), the error paths (`GapError` on interior gaps, weight
validation, bad switch values) and the NaN/inf conventions on degenerate input. The frozen results
are a regression net — they are generated from this package by `python -m tests.generate_expected`
and pin every number across releases and platforms.

Release history: [`CHANGELOG.md`](https://github.com/ekam-a3/paris/blob/main/CHANGELOG.md).

## Disclaimer

PARIS is a Python library for internal analytical, educational and research
use only. It does **not** provide investment, tax, legal or other professional
advice, and its output must **not** be relied upon for investment, trading,
reporting or any other decision-making. The library is provided "AS IS"
without warranty of any kind under the MIT License. Users are solely
responsible for independently verifying every number, method and convention
before use. PARIS is not a validated model and is not certified for regulated
use. See [`DISCLAIMER.md`](https://github.com/ekam-a3/paris/blob/main/DISCLAIMER.md)
for the full statement.
