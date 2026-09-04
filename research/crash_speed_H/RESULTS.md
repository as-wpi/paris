# Crash speed — test H (drawdown governor) — 2026-09-03

Pre-registration: `preregistration_H.json`. Two separate verdicts: crash-speed (the sequence's rule) and governor.

## Full sample (medians across 16 vehicles)

| Arm | Sharpe | CAGR | Martin | MaxDD | Mean weight | Turnover/yr |
|---|---:|---:|---:|---:|---:|---:|
| incumbent | 0.43 | 6.9% | 0.63 | -26.8% | 0.43 | 2.4 |
| H_parity | 0.43 | 4.4% | 1.05 | -12.8% | 0.21 | 1.4 |
| H_full | 0.44 | 6.0% | 0.57 | -24.2% | 0.38 | 3.0 |
| buy_hold | 0.42 | 8.7% | 0.49 | -63.3% | 1.00 | 0.0 |

## Episodes — median MaxDD by arm

| Episode | incumbent | H_parity | H_full |
|---|---:|---:|---:|
| E1987_crash | -6.1% | -2.7% | -5.9% |
| E1998_ltcm | -7.9% | -3.8% | -7.5% |
| X_2020_covid | -11.9% | -5.7% | -10.7% |
| E_bear_2018q4 | -6.1% | -2.9% | -5.7% |
| E2007_09_gfc | -8.4% | -3.8% | -8.0% |
| E2000_02_dotcom | -1.9% | -0.8% | -1.8% |
| B_bear_2022 | -1.3% | -0.7% | -1.3% |
| E2011_eu_us | -11.9% | -5.9% | -11.2% |
| K1995_99_bull | -10.3% | -4.5% | -9.6% |
| K2013_14_bull | -7.2% | -3.6% | -7.0% |
| A_bull_2021 | -9.5% | -4.7% | -8.1% |
| D_recent_2025plus | -9.3% | -4.2% | -8.8% |

## Verdicts


**H_parity — crash-speed: PASS; governor: PASS** — fast-crash ΔMaxDD +3.8 pts (bar +2.0); ΔSharpe -0.009 (CI [-0.017, +0.001]); ΔMartin +0.18 (CI [+0.14, +0.55]); full-sample MaxDD better in 16/16 (median +13.8 pts); turnover ×0.57; bleed pass.
- K1995_99_bull: bleed vs B&H +0.29 (arm) vs +0.29 (incumbent) ok
- K2013_14_bull: bleed vs B&H +0.46 (arm) vs +0.43 (incumbent) ok
- A_bull_2021: bleed vs B&H +0.51 (arm) vs +0.48 (incumbent) ok
- D_recent_2025plus: bleed vs B&H +0.16 (arm) vs +0.13 (incumbent) ok

**H_full — crash-speed: RETIRE; governor: RETIRE** — fast-crash ΔMaxDD +0.3 pts (bar +2.0); ΔSharpe -0.026 (CI [-0.047, -0.000]); ΔMartin -0.03 (CI [-0.11, +0.01]); full-sample MaxDD better in 16/16 (median +1.9 pts); turnover ×1.26; bleed pass.
- K1995_99_bull: bleed vs B&H +0.30 (arm) vs +0.29 (incumbent) ok
- K2013_14_bull: bleed vs B&H +0.50 (arm) vs +0.43 (incumbent) ok
- A_bull_2021: bleed vs B&H +0.57 (arm) vs +0.48 (incumbent) ok
- D_recent_2025plus: bleed vs B&H +0.22 (arm) vs +0.13 (incumbent) ok
