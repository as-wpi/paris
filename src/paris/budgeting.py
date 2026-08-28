"""Risk budgeting: Euler contributions of each asset to portfolio volatility, VaR and CVaR.

Conventions (Euler / component risk decomposition):
* Inputs are asset returns and **one** weight vector (a dated weight table raises; the weights are
  held constant over the sample, i.e. rebalanced every period). Weights must sum to 1 within
  ``1e-4`` unless ``normalize=True``.
* A contribution is ``w_i * d(risk)/d(w_i)``; by Euler's theorem the contributions sum to the
  portfolio risk. ``pct=True`` divides by that total.
* The covariance uses ``ddof=1``; third and fourth co-moments are population moments (``1/n``).
  With ``ddof=0`` every moment is a population moment and the total equals the single-series ``var`` / ``cvar`` of the portfolio series;
  with the default it equals them for ``ddof=1``, and for ``method="modified"`` only when
  ``ddof=0``.
* VaR / CVaR contributions carry the library's sign: they sum to a (negative) return, not to a
  positive loss; percentage contributions are sign-free.
* Historical CVaR contributions are the mean of ``w_i r_i`` over the periods whose constant-weight
  portfolio return lies below the historical VaR, so they sum to ``cvar(portfolio, "historical")``.
  Historical VaR has no Euler decomposition and is not offered.
* ``marginal_var`` / ``marginal_cvar`` use constant weights: the risk of the portfolio without asset *i* (remaining weights
  rescaled to the same total) minus the full portfolio's risk; they use the single-series
  conventions (population sigma, ``ddof=0``).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from paris._core import (
    ParisError,
    _cf_z,
    _es_series,
    _pdf,
    _ppf,
    _var_series,
    align,
    resolve_periods,
    to_weights,
)

__all__ = [
    "cvar_contribution",
    "marginal_cvar",
    "marginal_var",
    "var_contribution",
    "volatility_contribution",
]


# --------------------------------------------------------------------------- plumbing
def _prep(returns: Any, weights: Any, normalize: bool) -> tuple[pd.DataFrame, np.ndarray]:
    df, _, _ = align(returns)
    wt = to_weights(weights, df.columns, normalize=normalize)
    if len(wt) != 1:
        raise ParisError("risk contributions take a single weight vector, not a dated weight table")
    return df[wt.columns], wt.iloc[0].values.astype(float)


def _out(values: Any, index: pd.Index, pct: bool) -> pd.Series:
    s = pd.Series(np.asarray(values, dtype=float), index=index)
    if pct:
        return (s / s.sum()).rename("pct_contribution")
    return s.rename("contribution")


def _nan(index: pd.Index, pct: bool) -> pd.Series:
    return _out(np.full(len(index), np.nan), index, pct)


class _Moments:
    """Portfolio moments and their gradients in the weights."""

    def __init__(self, x: np.ndarray, w: np.ndarray, ddof: int):
        n = len(x)
        self.ok = n - ddof > 0 and n > 1
        if not self.ok:
            return
        self.mu = x.mean(axis=0)
        xc = x - self.mu
        sigma = xc.T @ xc / (n - ddof)
        y = xc @ w
        self.loc = float(self.mu @ w)
        self.pm2, self.dpm2 = float(w @ sigma @ w), 2.0 * sigma @ w
        self.pm3, self.dpm3 = float(np.mean(y ** 3)), 3.0 * xc.T @ (y ** 2) / n
        self.pm4, self.dpm4 = float(np.mean(y ** 4)), 4.0 * xc.T @ (y ** 3) / n
        self.ok = self.pm2 > 0

    def cornish_fisher(self, z: float):
        """h, dh/dw, skew, excess kurtosis and their gradients (Boudt, Peterson & Croux 2008)."""
        pm2, pm3, pm4, dpm2, dpm3, dpm4 = self.pm2, self.pm3, self.pm4, self.dpm2, self.dpm3, self.dpm4
        sd = math.sqrt(pm2)
        skew, exkurt = pm3 / pm2 ** 1.5, pm4 / pm2 ** 2 - 3.0
        dskew = (2.0 * pm2 ** 1.5 * dpm3 - 3.0 * pm3 * sd * dpm2) / (2.0 * pm2 ** 3)
        dexkurt = (pm2 * dpm4 - 2.0 * pm4 * dpm2) / pm2 ** 3
        h = _cf_z(z, skew, exkurt)
        dh = ((z ** 2 - 1.0) * dskew / 6.0 + (z ** 3 - 3.0 * z) * dexkurt / 24.0
              - (2.0 * z ** 3 - 5.0 * z) * skew * dskew / 18.0)
        return h, dh, skew, exkurt, dskew, dexkurt


# --------------------------------------------------------------------------- volatility
def volatility_contribution(returns: Any, weights: Any, periods_per_year: int | None = None, *,
                            annualize: bool = True, ddof: int = 1, pct: bool = False,
                            normalize: bool = False) -> pd.Series:
    """Contribution of each asset to portfolio volatility, ``w_i (Σw)_i / σ_p``; sums to the
    volatility of the constant-weight portfolio."""
    df, w = _prep(returns, weights, normalize)
    ppy = resolve_periods(df.index, periods_per_year) if annualize else 1
    m = _Moments(df.values, w, ddof)
    if not m.ok:
        return _nan(df.columns, pct)
    contrib = w * (0.5 * m.dpm2) / math.sqrt(m.pm2)
    return _out(contrib * math.sqrt(ppy), df.columns, pct)


# --------------------------------------------------------------------------- VaR / CVaR
def _var_gradient(m: _Moments, alpha: float, method: str) -> np.ndarray:
    """d(VaR as a return)/dw."""
    z = _ppf(alpha)
    sd = math.sqrt(m.pm2)
    if method == "gaussian":
        return m.mu + z * (0.5 * m.dpm2) / sd
    h, dh, *_ = m.cornish_fisher(z)                                     # modified (validated by the caller)
    return m.mu + h * m.dpm2 / (2.0 * sd) + sd * dh


def var_contribution(returns: Any, weights: Any, confidence: float = 0.95, method: str = "gaussian", *,
                     ddof: int = 1, pct: bool = False, normalize: bool = False) -> pd.Series:
    """Contribution of each asset to portfolio VaR (``gaussian`` / ``modified``); sums to the VaR
    of the constant-weight portfolio as a (negative) return."""
    df, w = _prep(returns, weights, normalize)
    if method not in ("gaussian", "modified"):
        raise ValueError("method must be 'gaussian' or 'modified' (historical VaR has no Euler decomposition)")
    m = _Moments(df.values, w, ddof)
    if not m.ok:
        return _nan(df.columns, pct)
    return _out(w * _var_gradient(m, 1.0 - confidence, method), df.columns, pct)


def _es_gradient(m: _Moments, alpha: float, method: str, operational: bool) -> np.ndarray:
    """d(CVaR as a return)/dw."""
    z = _ppf(alpha)
    sd = math.sqrt(m.pm2)
    if method == "gaussian":
        return m.mu - _pdf(z) / alpha * (0.5 * m.dpm2) / sd
    h, dh, skew, exkurt, dskew, dexkurt = m.cornish_fisher(z)
    e = _pdf(h) * (1.0 + h ** 3 * skew / 6.0 + (h ** 6 - 9.0 * h ** 4 + 9.0 * h ** 2 + 3.0) * skew ** 2 / 72.0
                   + (h ** 4 - 2.0 * h ** 2 - 1.0) * exkurt / 24.0) / alpha
    if operational and -e > h:                      # modified ES less severe than modified VaR: cap binds
        return m.mu + h * m.dpm2 / (2.0 * sd) + sd * dh
    return (m.mu - e * m.dpm2 / (2.0 * sd) + sd * e * h * dh
            - _pdf(h) * sd / alpha * (dh * (h ** 2 * skew / 2.0
                                            + (6.0 * h ** 5 - 36.0 * h ** 3 + 18.0 * h) * skew ** 2 / 72.0
                                            + (4.0 * h ** 3 - 4.0 * h) * exkurt / 24.0)
                                      + h ** 3 * dskew / 6.0
                                      + (h ** 6 - 9.0 * h ** 4 + 9.0 * h ** 2 + 3.0) * skew * dskew / 36.0
                                      + (h ** 4 - 2.0 * h ** 2 - 1.0) * dexkurt / 24.0))


def cvar_contribution(returns: Any, weights: Any, confidence: float = 0.95, method: str = "historical", *,
                      ddof: int = 1, pct: bool = False, operational: bool = True,
                      interpolation: str = "linear", normalize: bool = False) -> pd.Series:
    """Contribution of each asset to portfolio CVaR / expected shortfall (``historical``,
    ``gaussian``, ``modified``); sums to the CVaR of the constant-weight portfolio."""
    df, w = _prep(returns, weights, normalize)
    x, alpha = df.values, 1.0 - confidence
    if method == "historical":
        p = x @ w
        q = np.quantile(p, alpha, method=interpolation)
        tail = p < q
        rows = x[tail] if tail.any() else x[[int(np.argmin(p))]]   # no exceedance: CVaR = VaR
        return _out((rows * w).mean(axis=0), df.columns, pct)
    if method not in ("gaussian", "modified"):
        raise ValueError("method must be 'historical', 'gaussian' or 'modified'")
    m = _Moments(x, w, ddof)
    if not m.ok:
        return _nan(df.columns, pct)
    return _out(w * _es_gradient(m, alpha, method, operational), df.columns, pct)


# --------------------------------------------------------------------------- marginal
def _marginal(fn, returns, weights, normalize):
    df, w = _prep(returns, weights, normalize)
    x = df.values
    full = fn(pd.Series(x @ w, index=df.index))
    out = np.full(len(w), np.nan)
    for i in range(len(w)):
        keep = np.arange(len(w)) != i
        rest = w[keep].sum()
        if rest != 0:
            sub = x[:, keep] @ (w[keep] * w.sum() / rest)
            out[i] = fn(pd.Series(sub, index=df.index)) - full
    return pd.Series(out, index=df.columns, name="marginal")


def marginal_var(returns: Any, weights: Any, confidence: float = 0.95, method: str = "historical", *,
                 interpolation: str = "linear", ddof: int = 0, normalize: bool = False) -> pd.Series:
    """VaR of the portfolio without each asset (remaining weights rescaled) minus the full portfolio's
    VaR, both as returns (constant weights)."""
    return _marginal(lambda s: _var_series(s, confidence, method, interpolation, ddof),
                     returns, weights, normalize)


def marginal_cvar(returns: Any, weights: Any, confidence: float = 0.95, method: str = "historical", *,
                  interpolation: str = "linear", ddof: int = 0, operational: bool = True,
                  normalize: bool = False) -> pd.Series:
    """CVaR of the portfolio without each asset (remaining weights rescaled) minus the full
    portfolio's CVaR, both as returns."""
    return _marginal(lambda s: _es_series(s, confidence, method, interpolation, ddof, operational),
                     returns, weights, normalize)
