# Crash speed — tests A and B — 2026-09-03

Pre-registration (with the outcome-blind B amendment): `Strategy Research Memos/crash-speed-switch-vs-sizer/`.


## Test A, step 1 — where the drawdown accrues (incumbent stack; medians across vehicles)

| Episode | n | MaxDD | DD before sizer reaction | share before | days peak→trough | w at peak |
|---|---:|---:|---:|---:|---:|---:|
| E_bear_2018q4 | 16 | -6.1% | -6.1% | 100% | 16 | 0.93 |
| X_2020_covid | 16 | -11.9% | -8.4% | 82% | 80 | 0.69 |
| B_bear_2022 | 16 | -1.3% | -1.3% | 100% | 6 | 0.19 |
| E1973_oil_bear | 1 | -11.6% | -11.3% | 97% | 209 | 1.00 |
| E1987_crash | 1 | -6.1% | -5.3% | 87% | 35 | 0.86 |
| E1990_recession | 1 | -3.6% | -3.6% | 100% | 10 | 0.85 |
| E1997_asia | 1 | -7.0% | -7.0% | 100% | 15 | 0.70 |
| E1998_ltcm | 1 | -7.9% | -7.9% | 100% | 14 | 0.73 |
| E2000_02_dotcom | 2 | -1.9% | -0.7% | 61% | 10 | 0.21 |
| E2007_09_gfc | 6 | -8.4% | -7.0% | 99% | 64 | 0.61 |
| E2011_eu_us | 16 | -11.9% | -9.8% | 78% | 101 | 0.68 |
| E2015_16_china | 16 | -1.9% | -1.8% | 100% | 8 | 0.24 |
| K1995_99_bull | 1 | -10.3% | -6.2% | 60% | 65 | 0.68 |
| K2013_14_bull | 16 | -7.2% | -6.6% | 100% | 26 | 0.86 |
| A_bull_2021 | 16 | -9.5% | -9.1% | 100% | 25 | 0.87 |
| D_recent_2025plus | 16 | -9.3% | -9.3% | 100% | 55 | 0.65 |

## Test A, step 2 — lookahead bounds (FICTION): reachable share of MaxDD = (bound − actual)/|actual|, medians (positive = the bound loses less)

| Episode | both k=1 | switch k=1 | sizer k=1 | both k=5 | switch k=5 | sizer k=5 |
|---|---:|---:|---:|---:|---:|---:|
| E_bear_2018q4 | +15% | +5% | +15% | +33% | +5% | +33% |
| X_2020_covid | +20% | +1% | +17% | +48% | +19% | +27% |
| B_bear_2022 | +11% | +12% | +3% | +30% | +34% | +2% |
| E1973_oil_bear | +14% | +8% | -2% | +25% | +20% | +4% |
| E1987_crash | +1% | +5% | -2% | +21% | +13% | +12% |
| E1990_recession | +0% | +0% | +0% | +9% | +4% | +0% |
| E1997_asia | +29% | -2% | +34% | +38% | -2% | +37% |
| E1998_ltcm | +35% | +30% | +14% | +53% | +44% | +23% |
| E2000_02_dotcom | +63% | +64% | +0% | +72% | +73% | -5% |
| E2007_09_gfc | +7% | -1% | +5% | +19% | +6% | +14% |
| E2011_eu_us | +14% | +0% | +13% | +25% | +0% | +24% |
| E2015_16_china | +0% | +1% | -2% | +12% | +17% | -3% |
| K1995_99_bull | +9% | +10% | -0% | +38% | +30% | +2% |
| K2013_14_bull | +6% | +0% | +5% | +11% | +0% | +7% |
| A_bull_2021 | +14% | +4% | +8% | +27% | +13% | +11% |
| D_recent_2025plus | +6% | +0% | +5% | +16% | -0% | +16% |

**Fast-crash subset (E1987_crash, E1998_ltcm, X_2020_covid, E_bear_2018q4) medians:** both k=5 +41%, switch-only k=5 +16%, sizer-only k=5 +25%; both k=1 +18%.

**Test A verdict (pre-registered rule): MIXED — C–G proceed in the registered order.**

## Test B — execution timing (full sample, paired vs the incumbent; fast-crash MaxDD)

| Arm | ΔSharpe median (CI) | ΔMartin | fast-crash ΔMaxDD (pts) | turnover × |
|---|---:|---:|---:|---:|
| B: next-open | +0.002 ([-0.017, +0.007]) | +0.01 | +0.1 | 0.97 |
| B: t+2 next-close | -0.023 ([-0.045, -0.008]) | -0.03 | -0.1 | 2.54 |
| A: both k=1 | +0.159 ([+0.138, +0.180]) | +0.43 | +1.5 | 0.99 |

**Test B verdict (pre-registered rule): EXECUTION TIMING IS NOT A LEVER** — t+2 ΔSharpe -0.023, fast-crash ΔMaxDD -0.1 pts; next-open ΔSharpe +0.002, ΔMaxDD +0.1 pts.
