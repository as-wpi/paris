# Reproducing the 2026-09-02 walk-forward, and its numbers by paris version

Written 2026-10-28 after the in-house audit of that date found that the run's input file was never
recorded. The recorded tables in this folder are unchanged.

## Inputs

| Input | Recipe | Rows | SHA-256 |
|---|---|---|---|
| ETF closes, `input/etf_adjclose.csv` (git-ignored, vendor data) | `build_input.py`: `research/data_cache/<T>_adj.csv` (`fetch_fmp.py`, FMP dividend-adjusted, fetched 2026-09-03), column `adjClose`, cut at 2026-09-02, columns SPY, QQQ, IWM, EFA, EEM, TLT, GLD, XLK | 8,456 (1993-01-29 to 2026-09-02) | `19485b8f39df81c24bbc0fad0a5dc1879d2bf4a8c1c9c138bb27b3a4cf909921` |
| Fama-French daily three factors | Kenneth French Data Library, 202606 vintage (last row 20260630); the copy used is `~/Documents/Work/Personal_Trading_Strategies/Market Timing/shared_data/F-F_Research_Data_Factors_daily.CSV` | | `3d73942f7c0ba3accd021721876fef3059c6b02764a476dbdd9c311817f09710` |

`build_input.py` checks both checksums and exits non-zero on a mismatch. A data cache re-fetched
after a later dividend rescales the adjusted history, so its returns can differ in the last digits
and the tables may move; a later French vintage moves the MKT table.

## Command

```bash
python research/walkforward_2026-09-02/build_input.py --ff <F-F_Research_Data_Factors_daily.CSV>
python research/walkforward_jump.py --ff <F-F_Research_Data_Factors_daily.CSV> \
    --etf research/walkforward_2026-09-02/input/etf_adjclose.csv --out <a new folder>
python research/walkforward_2026-09-02/summarize.py <that folder>
```

One process runs MKT and the eight ETFs in turn, about 30 minutes. `summarize.py` prints the
`CROSS_ASSET.md` tables (mean over the eight ETFs, wins against buy-and-hold); on this folder it
reproduces all 605 numbers in `CROSS_ASSET.md`.

## Which code reproduces the recorded tables

The recorded tables were produced before paris 0.11.0 (`eda7312`, 2026-09-03 18:48), which seeds
every jump-model refit by `(random_state, refit position)` instead of one generator consumed
across refits (`CHANGELOG.md`, 0.11.0, **Changed**). Run on 2026-10-28 with the 0.10.0 source
(`git archive 2f1f9d4 src research/walkforward_jump.py`) and the inputs above:

- 17 of the 18 CSVs are byte-identical to the recorded ones, and `WALKFORWARD.md` matches apart from
  its date line.
- `TLT_table.csv` differs in 34 of 88 cells by at most 1.1e-5 (relative 4.6e-5), buy-and-hold
  included, so the difference is in the input, not the model; states, time in the market, turnover
  and the annual picks are identical, as is every number printed in `WALKFORWARD.md` and
  `CROSS_ASSET.md`. TLT is the only one of the eight funds with an ex-dividend date just before
  the fetches (2026-09-01), which fits a vendor re-adjustment between the recorded fetch
  (2026-09-02) and the cache (2026-09-03); the cause is not verified.

## Numbers by paris version

Same inputs and command, paris 0.11.0 (source unchanged from `eda7312` through `be0f8ff`), run
2026-10-28; full tables in `rerun_0.11.0_2026-10-28/` (`summarize.py rerun_0.11.0_2026-10-28`
prints the whole summary). Means over the eight ETFs; wins are against buy-and-hold, of 8.
Buy-and-hold rows are identical in both.

| Quantity | Recorded (before 0.11.0) | 0.11.0 |
|---|---|---|
| Trend λ5, Sharpe mean (buy-and-hold 0.461) / wins | 0.508 / 5 | 0.504 / 4 |
| Trend λ5, MaxDD mean (buy-and-hold −0.450) / wins | −0.246 / 8 | −0.245 / 8 |
| Trend λ5, Martin mean (buy-and-hold 0.751) / wins | 1.124 / 7 | 1.085 / 6 |
| Trend λ5, CAGR mean (buy-and-hold 0.096) | 0.080 | 0.079 |
| Gate, Sharpe mean / wins | 0.459 / 2 | 0.450 / 3 |
| Gate, MaxDD mean / wins | −0.219 / 8 | −0.224 / 8 |
| Gate, Martin mean / wins | 1.076 / 6 | 1.043 / 5 |
| Risk λ50, Sharpe mean / wins | 0.323 / 0 | 0.318 / 0 |
| Cells / joint K=2 / joint K=4, Sharpe mean | 0.303 / 0.412 / 0.356 | 0.297 / 0.401 / 0.334 |
| Joint K=4, turnover per year mean | 3.38 | 3.23 |
| Selected annually, Sharpe mean / wins | 0.365 / 2 | 0.307 / 1 |
| Selected annually, Martin mean / wins | 0.718 / 4 | 0.632 / 2 |
| CAGR mean, range of the fixed rules (gate to trend) | 6.5% to 8.0% | 6.4% to 7.9% |
| MKT (Fama-French), trend λ5 Sharpe | 0.762 | 0.782 |

What the record concludes holds under 0.11.0: trend λ5 is still the best single fixed rule on
Sharpe and Martin, the gate still has the shallowest mean drawdown of all rules, every rule's
mean CAGR is still below buy-and-hold's, and annual re-selection still scores below both
buy-and-hold and trend λ5 on mean Sharpe. Counts and means move: quote these with the paris
version, or quote the 0.11.0 column.
