# PARIS regimes vs the frozen rtl-screener rule — 2026-09-03

Pre-registration: `research/frozen_rule_benchmark/preregistration.json`. Universe SPY, QQQ, XLK, IWM, EFA, EEM, TLT, GLD; cost 10 bp one-way; cash = T-bill; exposure for T from information ≤ close T−1.


## FULL (full OOS; cross-ticker medians, n = 8 tickers)

| Arm | Sharpe | MaxDD | Martin | CAGR | Off-frac | Flips/yr |
|---|---:|---:|---:|---:|---:|---:|
| Buy & hold | 0.41 | -53.2% | 0.49 | 8.0% | 0.00 | 0.0 |
| Frozen S1 (SPY switch) | 0.12 | -37.7% | 0.14 | 2.5% | 0.38 | 13.3 |
| Frozen S2 (own x_raw>0) | 0.25 | -35.2% | 0.21 | 4.3% | 0.31 | 12.7 |
| Frozen S3 (own InnovLMAR<=1.25) | 0.16 | -46.5% | 0.11 | 3.1% | 0.23 | 19.1 |
| Frozen S1xS2 | 0.01 | -35.5% | 0.05 | 1.1% | 0.46 | 15.0 |
| Frozen stack S1xS2xS3 | -0.03 | -36.7% | 0.05 | 1.1% | 0.51 | 17.6 |
| PARIS trend λ5 | 0.38 | -29.0% | 0.59 | 6.1% | 0.41 | 1.7 |
| PARIS risk λ50 | 0.31 | -37.2% | 0.22 | 4.9% | 0.40 | 0.7 |
| PARIS gate | 0.36 | -23.3% | 0.54 | 5.2% | 0.49 | 2.1 |
| PARIS graded | 0.38 | -24.8% | 0.44 | 5.7% | 0.39 | 2.3 |
| PARIS trend λ5 x S1 (exploratory) | 0.02 | -23.8% | 0.08 | 1.2% | 0.58 | 8.9 |
| F1 risk+logFQ λ50 | 0.38 | -32.6% | 0.40 | 5.7% | 0.44 | 0.7 |
| F1 gate (trend λ5 x F1) | 0.36 | -23.4% | 0.48 | 5.1% | 0.50 | 2.1 |
| F2 risk+logInnov λ50 | 0.40 | -32.1% | 0.43 | 6.3% | 0.39 | 0.9 |
| F2 gate (trend λ5 x F2) | 0.43 | -22.7% | 0.63 | 6.1% | 0.48 | 2.2 |
| T1 trend+LLTdrift λ5 | 0.36 | -27.4% | 0.64 | 6.0% | 0.40 | 1.8 |
| T2 trend+TS λ5 | 0.36 | -27.4% | 0.64 | 6.0% | 0.40 | 1.8 |

## E_bear_2018q4 (bear_mandatory; cross-ticker medians, n = 8 tickers)

| Arm | Sharpe | MaxDD | Martin | CAGR | Off-frac | Flips/yr |
|---|---:|---:|---:|---:|---:|---:|
| Buy & hold | -0.36 | -17.8% | -0.80 | -6.6% | 0.00 | 0.0 |
| Frozen S1 (SPY switch) | 0.38 | -4.1% | 2.54 | 5.5% | 0.58 | 4.1 |
| Frozen S2 (own x_raw>0) | -0.35 | -4.7% | 0.88 | 0.3% | 0.60 | 11.4 |
| Frozen S3 (own InnovLMAR<=1.25) | -0.81 | -13.6% | -1.10 | -6.7% | 0.47 | 23.8 |
| Frozen S1xS2 | 0.30 | -2.7% | 2.11 | 4.2% | 0.71 | 4.1 |
| Frozen stack S1xS2xS3 | 0.26 | -2.6% | 2.55 | 3.6% | 0.71 | 8.3 |
| PARIS trend λ5 | -1.44 | -4.4% | 14.48 | -2.4% | 0.70 | 2.1 |
| PARIS risk λ50 | -1.10 | -9.3% | -0.80 | -6.0% | 0.57 | 1.0 |
| PARIS gate | -1.67 | -2.8% | 14.48 | -0.2% | 0.80 | 2.1 |
| PARIS graded | -1.44 | -5.8% | -1.55 | -6.8% | 0.66 | 2.1 |
| PARIS trend λ5 x S1 (exploratory) | -0.58 | -1.9% | 8.75 | 1.8% | 0.80 | 2.1 |
| F1 risk+logFQ λ50 | -0.10 | 0.0% | inf | 2.6% | 1.00 | 0.0 |
| F1 gate (trend λ5 x F1) | -1.44 | -2.8% | 14.48 | -0.2% | 0.85 | 2.1 |
| F2 risk+logInnov λ50 | -1.42 | -10.0% | -1.97 | -11.5% | 0.65 | 2.1 |
| F2 gate (trend λ5 x F2) | -1.61 | -4.4% | 14.30 | -3.3% | 0.77 | 2.1 |
| T1 trend+LLTdrift λ5 | -1.30 | -4.4% | 12.75 | -2.4% | 0.73 | 2.1 |
| T2 trend+TS λ5 | -1.30 | -4.4% | 12.75 | -2.4% | 0.73 | 2.1 |

## X_2020_covid (bear_mandatory; cross-ticker medians, n = 8 tickers)

| Arm | Sharpe | MaxDD | Martin | CAGR | Off-frac | Flips/yr |
|---|---:|---:|---:|---:|---:|---:|
| Buy & hold | 0.80 | -31.4% | 2.21 | 23.7% | 0.00 | 0.0 |
| Frozen S1 (SPY switch) | 1.18 | -11.9% | 3.87 | 23.0% | 0.29 | 9.8 |
| Frozen S2 (own x_raw>0) | 1.15 | -13.7% | 3.41 | 19.6% | 0.25 | 6.5 |
| Frozen S3 (own InnovLMAR<=1.25) | 1.46 | -10.0% | 7.23 | 30.8% | 0.25 | 9.2 |
| Frozen S1xS2 | 0.88 | -13.2% | 2.71 | 14.9% | 0.38 | 10.9 |
| Frozen stack S1xS2xS3 | 0.57 | -10.2% | 1.08 | 8.4% | 0.45 | 13.0 |
| PARIS trend λ5 | 0.96 | -17.2% | 2.06 | 19.5% | 0.18 | 2.2 |
| PARIS risk λ50 | -0.57 | -7.9% | -0.41 | -3.9% | 0.87 | 1.1 |
| PARIS gate | 0.70 | -11.3% | 1.85 | 8.4% | 0.55 | 2.2 |
| PARIS graded | 0.61 | -13.1% | 1.37 | 8.2% | 0.55 | 2.7 |
| PARIS trend λ5 x S1 (exploratory) | 0.79 | -13.9% | 2.21 | 10.8% | 0.35 | 9.8 |
| F1 risk+logFQ λ50 | -1.18 | -11.9% | -0.61 | -7.6% | 0.89 | 1.1 |
| F1 gate (trend λ5 x F1) | 0.63 | -13.4% | 1.52 | 8.4% | 0.58 | 2.2 |
| F2 risk+logInnov λ50 | -0.52 | -9.4% | -0.32 | -2.6% | 0.90 | 1.1 |
| F2 gate (trend λ5 x F2) | 0.59 | -12.9% | 1.83 | 7.1% | 0.53 | 2.7 |
| T1 trend+LLTdrift λ5 | 0.65 | -15.7% | 1.66 | 11.6% | 0.20 | 2.2 |
| T2 trend+TS λ5 | 0.65 | -15.7% | 1.66 | 11.6% | 0.20 | 2.2 |

## B_bear_2022 (bear_mandatory; cross-ticker medians, n = 8 tickers)

| Arm | Sharpe | MaxDD | Martin | CAGR | Off-frac | Flips/yr |
|---|---:|---:|---:|---:|---:|---:|
| Buy & hold | -0.82 | -30.7% | -1.19 | -20.6% | 0.00 | 0.0 |
| Frozen S1 (SPY switch) | -0.18 | -10.1% | -0.29 | -1.3% | 0.75 | 10.0 |
| Frozen S2 (own x_raw>0) | -1.36 | -13.9% | -1.35 | -10.6% | 0.81 | 11.0 |
| Frozen S3 (own InnovLMAR<=1.25) | -0.46 | -23.1% | -0.75 | -10.9% | 0.34 | 30.6 |
| Frozen S1xS2 | -0.60 | -5.2% | -1.15 | -3.2% | 0.90 | 6.0 |
| Frozen stack S1xS2xS3 | -0.73 | -5.5% | -1.17 | -3.9% | 0.91 | 8.0 |
| PARIS trend λ5 | -1.39 | -3.2% | inf | 0.2% | 0.98 | 0.5 |
| PARIS risk λ50 | -0.85 | 0.0% | inf | 1.3% | 1.00 | 0.0 |
| PARIS gate | -1.39 | -1.6% | inf | 0.5% | 0.99 | 0.5 |
| PARIS graded | -1.13 | -4.0% | -0.30 | -1.0% | 0.97 | 0.5 |
| PARIS trend λ5 x S1 (exploratory) | 1.54 | 0.0% | inf | 1.3% | 1.00 | 0.0 |
| F1 risk+logFQ λ50 | -0.70 | -0.0% | inf | 1.3% | 1.00 | 0.5 |
| F1 gate (trend λ5 x F1) | -1.39 | -1.6% | inf | 0.8% | 0.99 | 0.5 |
| F2 risk+logInnov λ50 | -0.48 | -0.1% | inf | 1.3% | 1.00 | 0.0 |
| F2 gate (trend λ5 x F2) | -1.39 | -1.6% | inf | 0.8% | 0.99 | 0.5 |
| T1 trend+LLTdrift λ5 | -1.65 | -6.1% | -0.38 | -2.2% | 0.95 | 1.5 |
| T2 trend+TS λ5 | -1.65 | -6.1% | -0.38 | -2.2% | 0.95 | 1.5 |

## K_bull_2013_14 (bull; cross-ticker medians, n = 8 tickers)

| Arm | Sharpe | MaxDD | Martin | CAGR | Off-frac | Flips/yr |
|---|---:|---:|---:|---:|---:|---:|
| Buy & hold | 0.93 | -13.3% | 3.59 | 13.7% | 0.00 | 0.0 |
| Frozen S1 (SPY switch) | -0.04 | -16.8% | -0.12 | -1.4% | 0.31 | 12.5 |
| Frozen S2 (own x_raw>0) | 0.99 | -8.7% | 2.16 | 10.3% | 0.18 | 7.0 |
| Frozen S3 (own InnovLMAR<=1.25) | 0.45 | -14.1% | 0.87 | 4.6% | 0.24 | 17.2 |
| Frozen S1xS2 | -0.04 | -11.3% | -0.03 | -0.9% | 0.38 | 13.0 |
| Frozen stack S1xS2xS3 | -0.08 | -11.7% | -0.13 | -1.4% | 0.45 | 16.2 |
| PARIS trend λ5 | 1.05 | -7.3% | 4.48 | 11.4% | 0.27 | 0.8 |
| PARIS risk λ50 | 0.93 | -13.3% | 3.59 | 13.7% | 0.00 | 0.0 |
| PARIS gate | 1.05 | -7.3% | 4.48 | 11.4% | 0.27 | 0.8 |
| PARIS graded | 0.99 | -9.5% | 3.30 | 12.9% | 0.14 | 1.0 |
| PARIS trend λ5 x S1 (exploratory) | 0.20 | -9.2% | 0.19 | 0.6% | 0.48 | 9.5 |
| F1 risk+logFQ λ50 | 0.93 | -13.3% | 3.59 | 13.7% | 0.00 | 0.0 |
| F1 gate (trend λ5 x F1) | 1.05 | -7.3% | 4.48 | 11.4% | 0.27 | 0.8 |
| F2 risk+logInnov λ50 | 0.93 | -13.3% | 3.59 | 13.7% | 0.00 | 0.0 |
| F2 gate (trend λ5 x F2) | 1.05 | -7.3% | 4.48 | 11.4% | 0.27 | 0.8 |
| T1 trend+LLTdrift λ5 | 1.05 | -7.3% | 4.54 | 11.3% | 0.27 | 0.8 |
| T2 trend+TS λ5 | 1.05 | -7.3% | 4.54 | 11.3% | 0.27 | 0.8 |

## A_bull_2021 (bull; cross-ticker medians, n = 8 tickers)

| Arm | Sharpe | MaxDD | Martin | CAGR | Off-frac | Flips/yr |
|---|---:|---:|---:|---:|---:|---:|
| Buy & hold | 0.82 | -11.6% | 3.83 | 13.0% | 0.00 | 0.0 |
| Frozen S1 (SPY switch) | -0.12 | -12.5% | -0.41 | -2.0% | 0.26 | 13.0 |
| Frozen S2 (own x_raw>0) | 0.41 | -10.7% | 1.48 | 4.1% | 0.12 | 10.5 |
| Frozen S3 (own InnovLMAR<=1.25) | 0.18 | -12.3% | 0.40 | 2.2% | 0.20 | 22.0 |
| Frozen S1xS2 | 0.24 | -10.5% | 1.07 | 1.6% | 0.29 | 14.0 |
| Frozen stack S1xS2xS3 | 0.06 | -10.2% | 0.18 | -0.1% | 0.38 | 21.5 |
| PARIS trend λ5 | 0.80 | -10.0% | 3.34 | 12.8% | 0.09 | 1.0 |
| PARIS risk λ50 | -0.12 | 0.0% | -0.40 | 0.0% | 1.00 | 0.0 |
| PARIS gate | 0.72 | -6.2% | 3.18 | 6.8% | 0.54 | 1.0 |
| PARIS graded | 0.72 | -6.3% | 3.18 | 6.8% | 0.54 | 1.5 |
| PARIS trend λ5 x S1 (exploratory) | -0.18 | -10.5% | -0.54 | -3.7% | 0.26 | 13.0 |
| F1 risk+logFQ λ50 | 0.07 | -3.3% | 0.08 | 0.0% | 0.88 | 0.5 |
| F1 gate (trend λ5 x F1) | 0.52 | -6.6% | 2.00 | 5.1% | 0.48 | 2.0 |
| F2 risk+logInnov λ50 | 0.06 | -4.2% | 0.06 | 0.0% | 0.92 | 0.5 |
| F2 gate (trend λ5 x F2) | 0.72 | -5.9% | 3.11 | 6.5% | 0.53 | 2.0 |
| T1 trend+LLTdrift λ5 | 0.78 | -10.0% | 3.59 | 11.9% | 0.07 | 1.0 |
| T2 trend+TS λ5 | 0.78 | -10.0% | 3.59 | 11.9% | 0.07 | 1.0 |

## C_mixed_2023_24 (bull; cross-ticker medians, n = 8 tickers)

| Arm | Sharpe | MaxDD | Martin | CAGR | Off-frac | Flips/yr |
|---|---:|---:|---:|---:|---:|---:|
| Buy & hold | 0.74 | -13.5% | 3.78 | 16.8% | 0.00 | 0.0 |
| Frozen S1 (SPY switch) | 0.21 | -11.8% | 1.44 | 7.5% | 0.32 | 9.0 |
| Frozen S2 (own x_raw>0) | 0.49 | -13.0% | 2.00 | 11.8% | 0.16 | 12.3 |
| Frozen S3 (own InnovLMAR<=1.25) | 0.48 | -12.8% | 1.98 | 11.8% | 0.21 | 20.1 |
| Frozen S1xS2 | 0.08 | -12.6% | 1.00 | 5.4% | 0.38 | 13.1 |
| Frozen stack S1xS2xS3 | -0.08 | -13.3% | 0.73 | 4.0% | 0.47 | 22.1 |
| PARIS trend λ5 | 0.21 | -10.9% | 1.77 | 7.0% | 0.39 | 2.5 |
| PARIS risk λ50 | 0.48 | -13.1% | 2.97 | 12.6% | 0.27 | 0.5 |
| PARIS gate | 0.25 | -10.9% | 1.75 | 8.2% | 0.41 | 2.8 |
| PARIS graded | 0.39 | -10.9% | 2.60 | 10.5% | 0.29 | 2.8 |
| PARIS trend λ5 x S1 (exploratory) | -0.30 | -10.6% | 0.27 | 1.6% | 0.56 | 8.0 |
| F1 risk+logFQ λ50 | 0.36 | -13.1% | 1.75 | 10.2% | 0.27 | 0.5 |
| F1 gate (trend λ5 x F1) | 0.17 | -10.9% | 1.66 | 6.7% | 0.40 | 2.8 |
| F2 risk+logInnov λ50 | 0.36 | -13.1% | 1.80 | 10.0% | 0.25 | 0.5 |
| F2 gate (trend λ5 x F2) | 0.16 | -10.9% | 1.75 | 6.6% | 0.40 | 2.8 |
| T1 trend+LLTdrift λ5 | 0.27 | -10.9% | 2.73 | 7.7% | 0.38 | 2.3 |
| T2 trend+TS λ5 | 0.27 | -10.9% | 2.73 | 7.7% | 0.38 | 2.3 |

## D_recent_2025plus (bull; cross-ticker medians, n = 8 tickers)

| Arm | Sharpe | MaxDD | Martin | CAGR | Off-frac | Flips/yr |
|---|---:|---:|---:|---:|---:|---:|
| Buy & hold | 0.90 | -20.8% | 4.28 | 24.8% | 0.00 | 0.0 |
| Frozen S1 (SPY switch) | 0.34 | -13.6% | 1.29 | 8.7% | 0.32 | 12.6 |
| Frozen S2 (own x_raw>0) | 0.71 | -12.8% | 3.06 | 16.2% | 0.17 | 9.6 |
| Frozen S3 (own InnovLMAR<=1.25) | 0.33 | -17.2% | 1.20 | 9.4% | 0.23 | 17.7 |
| Frozen S1xS2 | 0.25 | -13.6% | 1.08 | 7.2% | 0.34 | 13.8 |
| Frozen stack S1xS2xS3 | 0.36 | -14.6% | 1.48 | 8.2% | 0.43 | 17.4 |
| PARIS trend λ5 | 1.04 | -13.4% | 4.73 | 24.2% | 0.09 | 1.2 |
| PARIS risk λ50 | 0.51 | -16.9% | 1.63 | 12.9% | 0.26 | 1.8 |
| PARIS gate | 0.92 | -12.4% | 4.67 | 20.2% | 0.23 | 2.7 |
| PARIS graded | 0.82 | -13.8% | 3.84 | 20.1% | 0.20 | 3.0 |
| PARIS trend λ5 x S1 (exploratory) | 0.30 | -14.4% | 1.31 | 8.0% | 0.33 | 12.6 |
| F1 risk+logFQ λ50 | 0.51 | -16.9% | 1.72 | 12.6% | 0.30 | 1.2 |
| F1 gate (trend λ5 x F1) | 0.85 | -12.4% | 4.38 | 17.5% | 0.25 | 2.1 |
| F2 risk+logInnov λ50 | 0.80 | -15.5% | 2.80 | 19.0% | 0.27 | 1.8 |
| F2 gate (trend λ5 x F2) | 0.99 | -12.4% | 5.11 | 20.2% | 0.23 | 2.7 |
| T1 trend+LLTdrift λ5 | 0.97 | -13.0% | 4.97 | 21.1% | 0.10 | 1.8 |
| T2 trend+TS λ5 | 0.97 | -13.0% | 4.97 | 21.1% | 0.10 | 1.8 |

## Full-OOS Sharpe by ticker

| Ticker | Buy & hold | Frozen S1 (SPY switch) | Frozen S2 (own x_raw>0) | Frozen S3 (own InnovLMAR<=1.25) | Frozen S1xS2 | Frozen stack S1xS2xS3 | PARIS trend λ5 | PARIS risk λ50 | PARIS gate | PARIS graded | PARIS trend λ5 x S1 (exploratory) | F1 risk+logFQ λ50 | F1 gate (trend λ5 x F1) | F2 risk+logInnov λ50 | F2 gate (trend λ5 x F2) | T1 trend+LLTdrift λ5 | T2 trend+TS λ5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY | 0.42 | 0.16 | 0.32 | 0.17 | 0.08 | -0.04 | 0.63 | 0.44 | 0.65 | 0.61 | 0.25 | 0.58 | 0.64 | 0.41 | 0.62 | 0.62 | 0.62 |
| QQQ | 0.70 | 0.58 | 0.61 | 0.66 | 0.49 | 0.49 | 0.85 | 0.64 | 0.77 | 0.81 | 0.56 | 0.60 | 0.73 | 0.58 | 0.75 | 0.85 | 0.85 |
| XLK | 0.67 | 0.52 | 0.54 | 0.71 | 0.40 | 0.32 | 0.67 | 0.59 | 0.65 | 0.69 | 0.41 | 0.53 | 0.63 | 0.60 | 0.63 | 0.84 | 0.84 |
| IWM | 0.40 | 0.24 | 0.18 | 0.36 | -0.04 | -0.19 | 0.38 | 0.27 | 0.32 | 0.36 | 0.03 | 0.30 | 0.30 | 0.38 | 0.38 | 0.39 | 0.39 |
| EFA | 0.26 | 0.08 | 0.16 | 0.08 | 0.02 | -0.07 | 0.32 | 0.11 | 0.32 | 0.23 | -0.03 | 0.11 | 0.28 | 0.13 | 0.31 | 0.30 | 0.30 |
| EEM | 0.37 | -0.10 | 0.16 | 0.16 | -0.05 | 0.06 | 0.29 | 0.15 | 0.26 | 0.24 | -0.07 | 0.10 | 0.20 | 0.19 | 0.27 | 0.33 | 0.33 |
| TLT | 0.14 | -0.03 | -0.05 | -0.06 | -0.30 | -0.44 | 0.16 | 0.05 | 0.03 | 0.14 | -0.11 | 0.12 | 0.09 | 0.12 | 0.09 | 0.11 | 0.11 |
| GLD | 0.41 | -0.10 | 0.33 | 0.12 | 0.00 | -0.02 | 0.37 | 0.34 | 0.41 | 0.41 | 0.00 | 0.46 | 0.43 | 0.53 | 0.48 | 0.27 | 0.27 |

## Pre-registered decision rule


**PARIS trend λ5 vs Frozen stack S1xS2xS3: LOSES** — median ΔSharpe +0.39 (95% CI [+0.35, +0.58]), wins 8/8, median ΔMartin +0.61; bear folds FAIL; false-bear ceiling FAIL.
- E_bear_2018q4: MaxDD -4.4% vs stack -2.6% FAIL; ΔSharpe -1.71
- X_2020_covid: MaxDD -17.2% vs stack -10.2% FAIL; ΔSharpe +0.39
- B_bear_2022: MaxDD -3.2% vs stack -5.5% ok; ΔSharpe -0.65
- K_bull_2013_14: off-frac 0.27 vs stack 0.45 ok
- A_bull_2021: off-frac 0.09 vs stack 0.38 ok
- C_mixed_2023_24: off-frac 0.39 vs stack 0.47 FAIL
- D_recent_2025plus: off-frac 0.09 vs stack 0.43 ok

**PARIS risk λ50 vs Frozen stack S1xS2xS3: LOSES** — median ΔSharpe +0.31 (95% CI [+0.17, +0.47]), wins 8/8, median ΔMartin +0.25; bear folds FAIL; false-bear ceiling FAIL.
- E_bear_2018q4: MaxDD -9.3% vs stack -2.6% FAIL; ΔSharpe -1.37
- X_2020_covid: MaxDD -7.9% vs stack -10.2% ok; ΔSharpe -1.14
- B_bear_2022: MaxDD 0.0% vs stack -5.5% ok; ΔSharpe -0.11
- K_bull_2013_14: off-frac 0.00 vs stack 0.45 ok
- A_bull_2021: off-frac 1.00 vs stack 0.38 FAIL
- C_mixed_2023_24: off-frac 0.27 vs stack 0.47 ok
- D_recent_2025plus: off-frac 0.26 vs stack 0.43 ok

**PARIS gate vs Frozen stack S1xS2xS3: LOSES** — median ΔSharpe +0.41 (95% CI [+0.28, +0.51]), wins 8/8, median ΔMartin +0.55; bear folds FAIL; false-bear ceiling FAIL.
- E_bear_2018q4: MaxDD -2.8% vs stack -2.6% ok; ΔSharpe -1.94
- X_2020_covid: MaxDD -11.3% vs stack -10.2% FAIL; ΔSharpe +0.13
- B_bear_2022: MaxDD -1.6% vs stack -5.5% ok; ΔSharpe -0.65
- K_bull_2013_14: off-frac 0.27 vs stack 0.45 ok
- A_bull_2021: off-frac 0.54 vs stack 0.38 FAIL
- C_mixed_2023_24: off-frac 0.41 vs stack 0.47 FAIL
- D_recent_2025plus: off-frac 0.23 vs stack 0.43 ok

**PARIS graded vs Frozen stack S1xS2xS3: LOSES** — median ΔSharpe +0.40 (95% CI [+0.30, +0.57]), wins 8/8, median ΔMartin +0.47; bear folds FAIL; false-bear ceiling FAIL.
- E_bear_2018q4: MaxDD -5.8% vs stack -2.6% FAIL; ΔSharpe -1.71
- X_2020_covid: MaxDD -13.1% vs stack -10.2% FAIL; ΔSharpe +0.04
- B_bear_2022: MaxDD -4.0% vs stack -5.5% ok; ΔSharpe -0.39
- K_bull_2013_14: off-frac 0.14 vs stack 0.45 ok
- A_bull_2021: off-frac 0.54 vs stack 0.38 FAIL
- C_mixed_2023_24: off-frac 0.29 vs stack 0.47 ok
- D_recent_2025plus: off-frac 0.20 vs stack 0.43 ok

**F1 risk+logFQ λ50 vs Frozen stack S1xS2xS3: LOSES** — median ΔSharpe +0.34 (95% CI [+0.11, +0.56]), wins 8/8, median ΔMartin +0.37; bear folds FAIL; false-bear ceiling FAIL.
- E_bear_2018q4: MaxDD 0.0% vs stack -2.6% ok; ΔSharpe -0.37
- X_2020_covid: MaxDD -11.9% vs stack -10.2% FAIL; ΔSharpe -1.76
- B_bear_2022: MaxDD -0.0% vs stack -5.5% ok; ΔSharpe +0.03
- K_bull_2013_14: off-frac 0.00 vs stack 0.45 ok
- A_bull_2021: off-frac 0.88 vs stack 0.38 FAIL
- C_mixed_2023_24: off-frac 0.27 vs stack 0.47 ok
- D_recent_2025plus: off-frac 0.30 vs stack 0.43 ok

**F1 gate (trend λ5 x F1) vs Frozen stack S1xS2xS3: LOSES** — median ΔSharpe +0.40 (95% CI [+0.24, +0.53]), wins 8/8, median ΔMartin +0.50; bear folds FAIL; false-bear ceiling FAIL.
- E_bear_2018q4: MaxDD -2.8% vs stack -2.6% ok; ΔSharpe -1.71
- X_2020_covid: MaxDD -13.4% vs stack -10.2% FAIL; ΔSharpe +0.06
- B_bear_2022: MaxDD -1.6% vs stack -5.5% ok; ΔSharpe -0.65
- K_bull_2013_14: off-frac 0.27 vs stack 0.45 ok
- A_bull_2021: off-frac 0.48 vs stack 0.38 FAIL
- C_mixed_2023_24: off-frac 0.40 vs stack 0.47 FAIL
- D_recent_2025plus: off-frac 0.25 vs stack 0.43 ok

**F2 risk+logInnov λ50 vs Frozen stack S1xS2xS3: LOSES** — median ΔSharpe +0.36 (95% CI [+0.15, +0.55]), wins 8/8, median ΔMartin +0.39; bear folds FAIL; false-bear ceiling FAIL.
- E_bear_2018q4: MaxDD -10.0% vs stack -2.6% FAIL; ΔSharpe -1.68
- X_2020_covid: MaxDD -9.4% vs stack -10.2% ok; ΔSharpe -1.09
- B_bear_2022: MaxDD -0.1% vs stack -5.5% ok; ΔSharpe +0.25
- K_bull_2013_14: off-frac 0.00 vs stack 0.45 ok
- A_bull_2021: off-frac 0.92 vs stack 0.38 FAIL
- C_mixed_2023_24: off-frac 0.25 vs stack 0.47 ok
- D_recent_2025plus: off-frac 0.27 vs stack 0.43 ok

**F2 gate (trend λ5 x F2) vs Frozen stack S1xS2xS3: LOSES** — median ΔSharpe +0.44 (95% CI [+0.27, +0.57]), wins 8/8, median ΔMartin +0.64; bear folds FAIL; false-bear ceiling FAIL.
- E_bear_2018q4: MaxDD -4.4% vs stack -2.6% FAIL; ΔSharpe -1.88
- X_2020_covid: MaxDD -12.9% vs stack -10.2% FAIL; ΔSharpe +0.02
- B_bear_2022: MaxDD -1.6% vs stack -5.5% ok; ΔSharpe -0.65
- K_bull_2013_14: off-frac 0.27 vs stack 0.45 ok
- A_bull_2021: off-frac 0.53 vs stack 0.38 FAIL
- C_mixed_2023_24: off-frac 0.40 vs stack 0.47 FAIL
- D_recent_2025plus: off-frac 0.23 vs stack 0.43 ok

**T1 trend+LLTdrift λ5 vs Frozen stack S1xS2xS3: LOSES** — median ΔSharpe +0.44 (95% CI [+0.32, +0.58]), wins 8/8, median ΔMartin +0.65; bear folds FAIL; false-bear ceiling FAIL.
- E_bear_2018q4: MaxDD -4.4% vs stack -2.6% FAIL; ΔSharpe -1.57
- X_2020_covid: MaxDD -15.7% vs stack -10.2% FAIL; ΔSharpe +0.08
- B_bear_2022: MaxDD -6.1% vs stack -5.5% ok; ΔSharpe -0.92
- K_bull_2013_14: off-frac 0.27 vs stack 0.45 ok
- A_bull_2021: off-frac 0.07 vs stack 0.38 ok
- C_mixed_2023_24: off-frac 0.38 vs stack 0.47 FAIL
- D_recent_2025plus: off-frac 0.10 vs stack 0.43 ok

**T2 trend+TS λ5 vs Frozen stack S1xS2xS3: LOSES** — median ΔSharpe +0.44 (95% CI [+0.32, +0.58]), wins 8/8, median ΔMartin +0.65; bear folds FAIL; false-bear ceiling FAIL.
- E_bear_2018q4: MaxDD -4.4% vs stack -2.6% FAIL; ΔSharpe -1.57
- X_2020_covid: MaxDD -15.7% vs stack -10.2% FAIL; ΔSharpe +0.08
- B_bear_2022: MaxDD -6.1% vs stack -5.5% ok; ΔSharpe -0.92
- K_bull_2013_14: off-frac 0.27 vs stack 0.45 ok
- A_bull_2021: off-frac 0.07 vs stack 0.38 ok
- C_mixed_2023_24: off-frac 0.38 vs stack 0.47 FAIL
- D_recent_2025plus: off-frac 0.10 vs stack 0.43 ok

## Attribution: US equity vs ex-US names (full-OOS median ΔSharpe vs stack)

| Arm | US (SPY, QQQ, XLK, IWM) | ex-US (EFA, EEM, TLT, GLD) |
|---|---:|---:|
| PARIS trend λ5 | +0.47 | +0.39 |
| PARIS risk λ50 | +0.36 | +0.27 |
| PARIS gate | +0.42 | +0.41 |
| PARIS graded | +0.46 | +0.36 |
| F1 risk+logFQ λ50 | +0.35 | +0.33 |
| F1 gate (trend λ5 x F1) | +0.40 | +0.40 |
| F2 risk+logInnov λ50 | +0.36 | +0.37 |
| F2 gate (trend λ5 x F2) | +0.44 | +0.44 |
| T1 trend+LLTdrift λ5 | +0.55 | +0.32 |
| T2 trend+TS λ5 | +0.55 | +0.32 |

## Reading of the results (written after the run; the rule above was frozen before it)

**Stage 1, by the letter of the pre-registration: every challenger LOSES.** Two clauses do it. (i) The mandatory-fold MaxDD gate: the frozen stack is out of the market 71% of 2018-Q4, 45% of 2020 and 91% of 2022, so its bear-fold drawdowns (−2.6%, −10.2%, −5.5%) are bought with absence, and any rule that participates at all fails "MaxDD no worse within 1 pt". (ii) The absolute 0.35 false-bear cap: the trend model was off 39% of 2023–24 (a genuine late re-entry after the 2022 bear; Sharpe 0.21 vs buy-and-hold 0.74), and the gate/graded rows were off 54% of 2021 because the risk model labelled all of 2021 risk-off on the US names. The comparator clause was mis-specified — a MaxDD-no-worse test against a rule with 51% time out of market and a full-sample Sharpe of −0.03 is not a test of the challenger — but it is recorded as written and is not re-scored here.

**Stage 1, substance (all full-OOS, paired across the eight tickers, 95% bootstrap CI):**

| Comparison | ΔSharpe median (CI), wins | ΔMartin | ΔMaxDD |
|---|---|---:|---:|
| PARIS trend λ5 vs frozen stack S1×S2×S3 | +0.39 (+0.35, +0.58), 8/8 | +0.61 | +8 pts |
| PARIS trend λ5 vs frozen S2 (own x_raw>0, like-for-like) | +0.18 (+0.13, +0.24), 8/8 | +0.41 (8/8) | +6 pts (7/8) |
| PARIS trend λ5 vs buy-and-hold | +0.01 (−0.04, +0.13), 4/8 | +0.17 (6/8) | +27 pts (8/8) |
| PARIS gate vs buy-and-hold | −0.01 (−0.09, +0.07), 3/8 | +0.12 (7/8) | +30 pts (8/8) |

The frozen stack flips 17.6 times a year (S1 alone 13.3, S3 alone 19.1) against 1.7 for the trend model; at 10 bp one-way that is not the story (≈35 bp/yr), the whipsaw itself is. The stack's edge is confined to the mandatory bear folds, where it is essentially cash. The attribution clause cannot fire as written: the deficit vs the stack is not concentrated in the ex-US names (median ΔSharpe +0.39 ex-US vs +0.47 US, both positive) nor in bear-fold Sharpe in a way features could fix — the two genuine weaknesses of the PARIS trend model are **speed** (2020: MaxDD −17% vs the stack's −10%, a daily-lagged 5-year regime is slow in a 23-day crash) and **re-entry** (2023). Neither stress detection nor volatility measurement addresses speed.

**The risk model's calibration, not its feature, is the problem.** For SPY, risk λ=50 was risk-off continuously 1998–2001, 2008–2010 and **2018-Q4 through 2022** (four years); the low-volatility centre of the 2016–2021 window sat at 7–8% annualised (the 2017 calm) and the switching threshold at 10–11%, below 2021's realised 12–15%. Standalone it loses to buy-and-hold on Sharpe in 8/8 names (median 0.31 vs 0.41) and wins only on MaxDD. This is the fixed-window anchoring problem discussed on 2026-09-03 (exponential forgetting / long-memory anchor / ensemble of memories), now with a measured cost.

**Stage 2 — the amended-in LLT features. All four RETIRE** (rule: beat both the frozen stack and the untouched PARIS model of the same side).

| Test | vs untouched PARIS model (ΔSharpe median (CI), wins; ΔMartin; ΔMaxDD) | Verdict |
|---|---|---|
| F1 log fit quality beside log vol (risk) | standalone +0.01 (−0.05, +0.11), 4/8; +0.06; +4.6 pts (5/8). As gate: −0.02 (−0.04, +0.01), 2/8; **Martin −0.06 (−0.13, −0.01)**; 2020 fold worse (Sharpe −1.18 vs −0.57, MaxDD −11.9% vs −7.9%) | RETIRE |
| F2 log InnovLMAR beside log vol (risk) | standalone +0.03 (−0.02, +0.11), 6/8; +0.04; +1.7 pts (6/8, CI (+0.00, +0.05)). As gate: +0.00 (−0.03, +0.06), 4/8; +0.01; +0.6 pt | RETIRE — nearest miss; the F2 gate row is the best line in the table (0.43 / −22.7% / 0.63) but the paired difference to the plain gate is zero |
| T1 LLT drift beside slow trailing mean (trend) | −0.01 (−0.04, +0.04), 4/8; −0.02; 0.0 (3/8) | RETIRE |
| T2 LLT trend strength beside slow (trend) | **identical to T1 to the last digit** | RETIRE |

**Why T1 ≡ T2.** The LLT slope variance from the frozen priors converges to a constant (1.2827×10⁻⁵ from about bar 100 onward, min = median = max thereafter), so `trend_strength = slope/√svar` is the drift times a constant, and the per-window standardisation erases the constant. Consequence for the screener's own record: the S1 clause `trend_strength < 0` is exactly `x_raw_llt < 0`, and the 0.75 / 2.0 "trend / strong" bands on TS are drift thresholds in disguise (≈ 0.68% and 1.8% annualised drift). Not a bug in either codebase; it is a property of a steady-state Kalman filter with fixed Q and R.

**Disposition.** No library default changes. The amendment produced four clean RETIREs and one structural fact (TS ≡ drift); the LLT family is closed as jump-model input on this evidence. The next admissible measurement test remains B1 (range-based volatility replacing close-to-close EWMA), but the risk-model diagnosis above says the calibration window, not the estimator, is the first-order problem — that is a design decision for the user, not a test to run unasked.
