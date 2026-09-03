# Crash speed — tests E, F, G — 2026-09-03

Pre-registration: `preregistration_EFG.json`. Ceiling from test A: switch-only k=5 bound = 16% of fast-crash MaxDD (~1 pt).

## Full sample (medians across 16 vehicles)

| Arm | Sharpe | Martin | MaxDD | Turnover/yr | Flips/yr |
|---|---:|---:|---:|---:|---:|
| incumbent | 0.43 | 0.63 | -26.8% | 2.4 | 1.7 |
| E_2s | 0.42 | 0.63 | -27.6% | 2.5 | 1.8 |
| E_3s | 0.43 | 0.63 | -26.8% | 2.4 | 1.7 |
| F_exit2 | 0.41 | 0.55 | -27.4% | 2.5 | 2.1 |
| F_exit1 | 0.41 | 0.55 | -27.7% | 2.7 | 2.6 |
| G_own | 0.45 | 0.70 | -26.5% | 3.9 |  |
| G_spy | 0.45 | 0.78 | -25.4% | 3.9 |  |

## Fast-crash and calm episodes — median MaxDD by arm

| Episode | incumbent | E_2s | E_3s | F_exit2 | F_exit1 | G_own | G_spy |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1987_crash | -6.1% | -6.1% | -6.1% | -6.3% | -6.7% | -6.1% | -6.1% |
| E1998_ltcm | -7.9% | -7.9% | -7.9% | -5.5% | -4.7% | -7.9% | -7.9% |
| X_2020_covid | -11.9% | -11.9% | -11.9% | -10.3% | -10.8% | -10.9% | -10.4% |
| E_bear_2018q4 | -6.1% | -6.1% | -6.1% | -5.5% | -5.4% | -5.7% | -5.9% |
| K1995_99_bull | -10.3% | -9.4% | -10.3% | -10.1% | -8.7% | -10.3% | -10.3% |
| K2013_14_bull | -7.2% | -7.2% | -7.2% | -7.0% | -7.0% | -7.1% | -7.1% |
| A_bull_2021 | -9.5% | -9.5% | -9.5% | -8.2% | -8.8% | -9.5% | -9.5% |
| D_recent_2025plus | -9.3% | -9.3% | -9.3% | -9.8% | -9.4% | -9.4% | -9.6% |

## Pre-registered decision rule


**E_2s: RETIRE** — fast-crash ΔMaxDD +0.0 pts (bar +2.0; ceiling ~+1.0); ΔSharpe +0.000 (CI [-0.013, +0.000], bar −0.02); turnover ×1.00 (bar 1.25); flips/yr 1.8 (bar 4); calm-control bleed pass.
- K1995_99_bull: bleed vs B&H +0.26 (arm) vs +0.29 (incumbent) ok
- K2013_14_bull: bleed vs B&H +0.43 (arm) vs +0.43 (incumbent) ok
- A_bull_2021: bleed vs B&H +0.48 (arm) vs +0.48 (incumbent) ok
- D_recent_2025plus: bleed vs B&H +0.13 (arm) vs +0.13 (incumbent) ok

**E_3s: RETIRE** — fast-crash ΔMaxDD +0.0 pts (bar +2.0; ceiling ~+1.0); ΔSharpe +0.000 (CI [+0.000, +0.000], bar −0.02); turnover ×1.00 (bar 1.25); flips/yr 1.7 (bar 4); calm-control bleed pass.
- K1995_99_bull: bleed vs B&H +0.29 (arm) vs +0.29 (incumbent) ok
- K2013_14_bull: bleed vs B&H +0.43 (arm) vs +0.43 (incumbent) ok
- A_bull_2021: bleed vs B&H +0.48 (arm) vs +0.48 (incumbent) ok
- D_recent_2025plus: bleed vs B&H +0.13 (arm) vs +0.13 (incumbent) ok

**F_exit2: RETIRE** — fast-crash ΔMaxDD +0.0 pts (bar +2.0; ceiling ~+1.0); ΔSharpe -0.013 (CI [-0.035, -0.006], bar −0.02); turnover ×1.04 (bar 1.25); flips/yr 2.1 (bar 4); calm-control bleed pass.
- K1995_99_bull: bleed vs B&H +0.30 (arm) vs +0.29 (incumbent) ok
- K2013_14_bull: bleed vs B&H +0.43 (arm) vs +0.43 (incumbent) ok
- A_bull_2021: bleed vs B&H +0.47 (arm) vs +0.48 (incumbent) ok
- D_recent_2025plus: bleed vs B&H +0.17 (arm) vs +0.13 (incumbent) ok

**F_exit1: RETIRE** — fast-crash ΔMaxDD -0.0 pts (bar +2.0; ceiling ~+1.0); ΔSharpe -0.028 (CI [-0.042, -0.013], bar −0.02); turnover ×1.14 (bar 1.25); flips/yr 2.6 (bar 4); calm-control bleed FAIL.
- K1995_99_bull: bleed vs B&H +0.47 (arm) vs +0.29 (incumbent) FAIL
- K2013_14_bull: bleed vs B&H +0.33 (arm) vs +0.43 (incumbent) ok
- A_bull_2021: bleed vs B&H +0.63 (arm) vs +0.48 (incumbent) FAIL
- D_recent_2025plus: bleed vs B&H +0.19 (arm) vs +0.13 (incumbent) ok

**G_own: RETIRE** — fast-crash ΔMaxDD -0.0 pts (bar +2.0; ceiling ~+1.0); ΔSharpe -0.004 (CI [-0.017, +0.008], bar −0.02); turnover ×1.59 (bar 1.25); flips/yr n/a (bar 4); calm-control bleed pass.
- K1995_99_bull: bleed vs B&H +0.34 (arm) vs +0.29 (incumbent) ok
- K2013_14_bull: bleed vs B&H +0.39 (arm) vs +0.43 (incumbent) ok
- A_bull_2021: bleed vs B&H +0.46 (arm) vs +0.48 (incumbent) ok
- D_recent_2025plus: bleed vs B&H +0.16 (arm) vs +0.13 (incumbent) ok

**G_spy: RETIRE** — fast-crash ΔMaxDD -0.0 pts (bar +2.0; ceiling ~+1.0); ΔSharpe -0.010 (CI [-0.018, +0.001], bar −0.02); turnover ×1.53 (bar 1.25); flips/yr n/a (bar 4); calm-control bleed FAIL.
- K1995_99_bull: bleed vs B&H +0.34 (arm) vs +0.29 (incumbent) ok
- K2013_14_bull: bleed vs B&H +0.43 (arm) vs +0.43 (incumbent) ok
- A_bull_2021: bleed vs B&H +0.48 (arm) vs +0.48 (incumbent) ok
- D_recent_2025plus: bleed vs B&H +0.26 (arm) vs +0.13 (incumbent) FAIL
