# Volatility estimators inside the vol-target loop — 2026-09-03

Pre-registration: `preregistration.json`. Gate = trend λ5 on the 1x, calibration **rolling**. 5+3 bp one-way, T-bill cash, sigma and gate at close T−1. Tracking metrics on the ungated loop; outcome metrics on the gated loop.


## P1 1x (target 10%; cross-vehicle medians)

| Estimator | Gate | RealVol | Track RMSE63 | Sharpe | Martin | MaxDD | CAGR | Turnover/yr | Mean w |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 EWMA c2c | none | 10.1% | 1.4% | 0.39 | 0.48 | -25.1% | 5.1% | 2.0 | 0.61 |
| E0 EWMA c2c | trend | 7.7% | 5.8% | 0.38 | 0.59 | -14.5% | 4.3% | 2.3 | 0.39 |
| E1 Yang-Zhang 21 | none | 10.5% | 1.6% | 0.39 | 0.48 | -26.8% | 5.3% | 2.1 | 0.63 |
| E1 Yang-Zhang 21 | trend | 8.0% | 5.8% | 0.37 | 0.60 | -14.6% | 4.3% | 2.3 | 0.41 |
| E2 Parkinson EWMA | none | 12.5% | 3.3% | 0.43 | 0.53 | -33.3% | 6.5% | 1.4 | 0.79 |
| E2 Parkinson EWMA | trend | 10.1% | 6.4% | 0.40 | 0.58 | -18.2% | 5.2% | 2.0 | 0.50 |
| E3 K-ATR fast | none | 12.1% | 2.6% | 0.33 | 0.38 | -35.3% | 5.1% | 8.9 | 0.77 |
| E3 K-ATR fast | trend | 9.8% | 6.1% | 0.33 | 0.43 | -17.9% | 4.3% | 6.3 | 0.50 |
| E4 Yang-Zhang EWMA | none | 10.1% | 1.7% | 0.41 | 0.53 | -26.3% | 5.5% | 1.6 | 0.61 |
| E4 Yang-Zhang EWMA | trend | 7.7% | 5.8% | 0.38 | 0.56 | -14.7% | 4.2% | 2.0 | 0.40 |

### Decision rule, P1 1x (paired vs E0)

**E1 Yang-Zhang 21: RETIRE** — ΔTrackRMSE +0.16% (CI [+0.07%, +0.24%]), better 0/8; ΔSharpe (gated) -0.03 (CI [-0.05, +0.01]); ΔMartin -0.04; turnover ×1.01.

**E2 Parkinson EWMA: RETIRE** — ΔTrackRMSE +2.03% (CI [+1.19%, +3.50%]), better 0/8; ΔSharpe (gated) -0.02 (CI [-0.04, -0.02]); ΔMartin +0.02; turnover ×0.90.

**E3 K-ATR fast: RETIRE** — ΔTrackRMSE +1.30% (CI [+0.87%, +2.39%]), better 0/8; ΔSharpe (gated) +0.01 (CI [-0.03, +0.04]); ΔMartin +0.07; turnover ×3.03.

**E4 Yang-Zhang EWMA: RETIRE** — ΔTrackRMSE +0.20% (CI [+0.18%, +0.32%]), better 0/8; ΔSharpe (gated) -0.01 (CI [-0.04, +0.02]); ΔMartin -0.00; turnover ×0.87.


## P2 3x (target 30%; cross-vehicle medians)

| Estimator | Gate | RealVol | Track RMSE63 | Sharpe | Martin | MaxDD | CAGR | Turnover/yr | Mean w |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 EWMA c2c | none | 30.1% | 4.5% | 0.42 | 0.43 | -51.1% | 9.9% | 2.9 | 0.60 |
| E0 EWMA c2c | trend | 22.7% | 17.6% | 0.41 | 0.51 | -41.9% | 8.3% | 2.8 | 0.43 |
| E1 Yang-Zhang 21 | none | 31.3% | 4.8% | 0.41 | 0.39 | -54.5% | 9.9% | 2.8 | 0.62 |
| E1 Yang-Zhang 21 | trend | 24.6% | 17.4% | 0.41 | 0.52 | -37.4% | 8.4% | 2.7 | 0.45 |
| E2 Parkinson EWMA | none | 36.8% | 9.2% | 0.46 | 0.45 | -61.4% | 12.0% | 1.9 | 0.82 |
| E2 Parkinson EWMA | trend | 28.7% | 18.9% | 0.43 | 0.51 | -47.1% | 9.5% | 2.2 | 0.52 |
| E3 K-ATR fast | none | 35.9% | 7.7% | 0.43 | 0.37 | -63.5% | 10.9% | 8.9 | 0.76 |
| E3 K-ATR fast | trend | 27.3% | 18.3% | 0.37 | 0.38 | -40.1% | 7.8% | 5.8 | 0.51 |
| E4 Yang-Zhang EWMA | none | 30.1% | 5.2% | 0.43 | 0.45 | -51.5% | 10.4% | 2.3 | 0.59 |
| E4 Yang-Zhang EWMA | trend | 23.5% | 17.6% | 0.42 | 0.54 | -37.2% | 8.4% | 2.6 | 0.43 |

### Decision rule, P2 3x (paired vs E0)

**E1 Yang-Zhang 21: RETIRE** — ΔTrackRMSE +0.48% (CI [+0.35%, +0.66%]), better 0/7; ΔSharpe (gated) -0.01 (CI [-0.07, +0.00]); ΔMartin -0.01; turnover ×0.99.

**E2 Parkinson EWMA: RETIRE** — ΔTrackRMSE +4.51% (CI [+3.57%, +7.18%]), better 0/7; ΔSharpe (gated) -0.01 (CI [-0.02, +0.05]); ΔMartin +0.01; turnover ×0.89.

**E3 K-ATR fast: RETIRE** — ΔTrackRMSE +2.93% (CI [+1.68%, +4.39%]), better 0/7; ΔSharpe (gated) +0.00 (CI [-0.07, +0.04]); ΔMartin +0.03; turnover ×2.33.

**E4 Yang-Zhang EWMA: RETIRE** — ΔTrackRMSE +0.70% (CI [+0.49%, +0.74%]), better 0/7; ΔSharpe (gated) -0.01 (CI [-0.01, +0.00]); ΔMartin +0.01; turnover ×0.86.

