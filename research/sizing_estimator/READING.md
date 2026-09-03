# Readings (written after each run; rules frozen before)

## E0–E3 run

**All three range-based estimators RETIRE, 0/8 and 0/7 on tracking.** The incumbent EWMA of close-to-close returns is the best tracker on every vehicle in both panels: rolling-63 RMSE around target 1.4% (1x) and 4.5% (3x) against 1.6% / 4.8% for Yang–Zhang, 3.3% / 9.2% for Parkinson, 2.6% / 7.7% for the K-ATR arm. Sharpe and Martin are a wash everywhere (|ΔSharpe| ≤ 0.03), so nothing was lost by the incumbent either.

**Why the literature efficiency did not show up.** (i) Yang–Zhang is unbiased and nearly matches EWMA, but a 21-day flat window is a worse temporal filter for a control loop than an exponential one with ~33 effective days — the estimator was better, the window was worse, and the loop only sees the product. (ii) Parkinson is biased low under discrete sampling (the intraday high and low understate the continuous extremes), so the loop over-levers: realised vol 12.5% against a 10% target on the 1x panel, 36.8% against 30% on the 3x. (iii) The K-ATR arm inherits a conversion constant (E[H−L] = 1.596σ) that is only right for a driftless diffusion without gaps, and the fast LLT reacts in 5–10 bars, which triples turnover (×3.0 / ×2.3) for no tracking gain. A range estimator is a better *daily* measurement; the vol-target needs a better *filtered* one, and the RiskMetrics filter is already good at that.

**Disposition.** EWMA λ=0.94 stays the sizing estimator; this is the third head-to-head it has won (EGARCH 2026-08-31, GJR term structure, now three range estimators). Range-based volatility is closed as a sizing input on this evidence. An EWMA-weighted Yang–Zhang (same filter, better daily measurement) is the one untested combination that the reading above predicts could match or edge the incumbent; it is a small spec, not run here.

## E4 run (EWMA-weighted Yang–Zhang; amendment frozen before the run)

**RETIRE, 0/8 and 0/7 on tracking.** With the incumbent's own filter, Yang–Zhang is unbiased (realised vol 10.1% and 30.1% against 10% and 30% targets, exactly matching E0) and trades 13–14% less, but its rolling-63 tracking RMSE is worse on every vehicle: 1.7% vs 1.4% (1x), 5.2% vs 4.5% (3x), CIs excluding zero. Sharpe and Martin are a wash (|Δ| ≤ 0.01).

**What this settles.** The hypothesis was that the flat window, not the estimator, handicapped Yang–Zhang; that is now rejected, since giving it the exponential filter did not help. The residual tracking loss is the estimator's: the overnight and open-based components add day-to-day noise that the close-to-close filter does not carry, and for a loop whose target is the realised close-to-close volatility of the sized strategy, the close-to-close estimator is measuring the right object. The one point in E4's favour, lower turnover at equal Sharpe, is real but small and was not the registered criterion. The estimator thread is closed: EWMA λ=0.94 close-to-close, four head-to-heads, four wins.
