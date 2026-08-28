"""Risk-adjusted performance ratios.

Conventions:
* Excess returns are arithmetic (``r - rf``). ``rf`` is an annual rate
  (de-annualised geometrically per period) or an aligned per-period Series.
* ``sharpe`` = ``mean(excess) * ppy / (sd(excess) * sqrt(ppy))``; ``geometric=True`` uses the
  annualised geometric excess return in the numerator instead.
* ``mar`` (minimum acceptable return) is a PER-PERIOD threshold.
* ``sortino`` is annualised by ``sqrt(ppy)`` by default; ``annualize=False`` returns the
  per-period ratio.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from paris._core import (
    _cdf,
    _downside,
    _es_series,
    _std_moment,
    _var_series,
    annualize_return,
    prepare,
    result,
    rf_annual,
)

__all__ = [
    "adjusted_sharpe",
    "adjusted_sortino",
    "autocorr_penalty",
    "bernardo_ledoit_ratio",
    "common_sense_ratio",
    "cpc_index",
    "d_ratio",
    "gain_to_pain_ratio",
    "kappa",
    "kelly_criterion",
    "kelly_ratio",
    "omega",
    "payoff_ratio",
    "probabilistic_sharpe",
    "profit_factor",
    "prospect_ratio",
    "risk_of_ruin",
    "risk_return_ratio",
    "serenity_index",
    "sharpe",
    "skewness_kurtosis_ratio",
    "sortino",
    "upside_potential_ratio",
    "volatility_skewness",
    "win_loss_ratio",
]


# --------------------------------------------------------------------------- Sharpe family
def autocorr_penalty(returns: Any) -> Any:
    """Autocorrelation penalty ``sqrt(1 + 2*sum((n-k)/n * rho_k))`` used by the 'smart' ratios."""
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        x = s.values
        n = len(x)
        coef = np.abs(np.corrcoef(x[:-1], x[1:])[0, 1])
        corr = [((n - k) / n) * coef**k for k in range(1, n)]
        return float(np.sqrt(1 + 2 * np.sum(corr)))
    return result(p, fn)


def sharpe(returns: Any, rf: Any = 0.0, periods_per_year: int | None = None, annualize: bool = True,
           geometric: bool = False, compounding: bool = True, smart: bool = False,
           risk: str = "std", confidence: float = 0.95, ddof: int = 1) -> Any:
    """Sharpe ratio of excess returns.

    ``risk``: ``"std"`` (default), ``"var"`` or ``"cvar"`` (modified Cornish-Fisher). The tail-risk
    ratios are always per-period (mean excess / tail measure); ``annualize`` applies to
    ``risk="std"`` only.
    ``annualize=False`` returns the per-period ratio (mean excess / risk).
    ``smart=True`` divides by :func:`autocorr_penalty`.
    """
    p = prepare(returns, rf=rf, periods_per_year=periods_per_year, compounding=compounding)
    ppy = p.ppy if (annualize and risk == "std") else 1

    def fn(x: pd.Series) -> float:
        ret = annualize_return(x, ppy, geometric)
        if risk == "std":
            den = x.std(ddof=ddof) * np.sqrt(ppy)
        elif risk == "var":
            den = -_var_series(x, confidence, "modified", "linear", 0)
        elif risk == "cvar":
            den = -_es_series(x, confidence, "modified", "linear", 0)
        else:
            raise ValueError("risk must be 'std', 'var' or 'cvar'")
        out = ret / den
        if smart:
            out /= autocorr_penalty(x)
        return float(out)
    return result(p, fn, p.excess)


def probabilistic_sharpe(returns: Any, rf: Any = 0.0, benchmark_sharpe: float = 0.0,
                         periods_per_year: int | None = None, compounding: bool = True,
                         method: str = "sample") -> Any:
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado 2012): probability that the true
    per-period Sharpe exceeds ``benchmark_sharpe`` (per-period):
    ``Phi((SR - SR*) / sqrt((1 - g3*SR + (g4-1)/4*SR^2) / (n-1)))`` with skewness ``g3`` and
    non-excess kurtosis ``g4``. ``method="sample"`` (default) uses the bias-corrected sample
    estimators (pandas ``.skew()`` / ``.kurt()``); ``"moment"`` uses population moments."""
    p = prepare(returns, rf=rf, periods_per_year=periods_per_year, compounding=compounding)

    def fn(x: pd.Series) -> float:
        n = len(x)
        if n < 4 or np.ptp(x.values) == 0.0:  # no dispersion / no sample moments: undefined
            return float("nan")
        sr = x.mean() / x.std(ddof=1)
        if method == "sample":
            g3, g4 = x.skew(), x.kurt() + 3  # bias-corrected (Fisher) sample estimates
        elif method == "moment":
            g3, g4 = _skew(x), _kurt(x)  # population moments
        else:
            raise ValueError("method must be 'sample' or 'moment'")
        sigma = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr**2) / (n - 1))
        return float(_cdf((sr - benchmark_sharpe) / sigma))
    return result(p, fn, p.excess)


def adjusted_sharpe(returns: Any, rf: Any = 0.0, periods_per_year: int | None = None,
                    compounding: bool = True) -> Any:
    """Pezier & White adjusted Sharpe ``SR*(1 + S/6*SR - (K-3)/24*SR^2)`` with annualised
    (arithmetic) SR and population skewness/kurtosis."""
    p = prepare(returns, rf=rf, periods_per_year=periods_per_year, compounding=compounding)

    def fn(x: pd.Series) -> float:
        sr = annualize_return(x, p.ppy, False) / (x.std(ddof=1) * np.sqrt(p.ppy))
        return float(sr * (1 + sr * _skew(x) / 6 - (_kurt(x) - 3) / 24 * sr**2))
    return result(p, fn, p.excess)


def _skew(x: pd.Series) -> float:
    return _std_moment(x.values, 3)  # NaN for a constant series


def _kurt(x: pd.Series) -> float:
    return _std_moment(x.values, 4)


# --------------------------------------------------------------------------- downside ratios
def _mar(mar: Any, index: pd.Index) -> pd.Series:
    if np.isscalar(mar):
        return pd.Series(float(mar), index=index)
    return mar.reindex(index) if isinstance(mar, pd.Series) else pd.Series(np.asarray(mar, float), index=index)


def sortino(returns: Any, mar: Any = 0.0, periods_per_year: int | None = None, annualize: bool = True,
            method: str = "full", smart: bool = False) -> Any:
    """``mean(r - mar) / downside_deviation(mar)``, times ``sqrt(ppy)`` when annualised.

    ``annualize=False`` returns the per-period ratio; the default is annualised, with the
    full-sample downside deviation and MAR=0.
    """
    p = prepare(returns, periods_per_year=periods_per_year if annualize else 1)
    m = _mar(mar, p.returns.index)

    def fn(s: pd.Series) -> float:
        out = float((s - m).mean() / _downside(s, m, method)) * np.sqrt(p.ppy if annualize else 1)
        return out / autocorr_penalty(s) if smart else out
    return result(p, fn)


def adjusted_sortino(returns: Any, mar: Any = 0.0, periods_per_year: int | None = None,
                     annualize: bool = True) -> Any:
    """Sortino / sqrt(2), comparable in scale to Sharpe (adjusted Sortino)."""
    return sortino(returns, mar, periods_per_year, annualize) / np.sqrt(2)


def omega(returns: Any, mar: Any = 0.0) -> Any:
    """Omega ratio ``mean(max(r-mar,0)) / mean(max(mar-r,0))`` ."""
    p = prepare(returns, periods_per_year=1)
    m = _mar(mar, p.returns.index)

    def fn(s: pd.Series) -> float:
        d = s.values - m.values
        return float(np.maximum(d, 0).mean() / np.maximum(-d, 0).mean())
    return result(p, fn)


def kappa(returns: Any, mar: Any = 0.0, order: float = 3) -> Any:
    """Kaplan-Knowles Kappa ``(mean(r) - mar) / (mean(max(mar-r,0)^order))^(1/order)``; ``order=2``
    is the (un-annualised) Sortino ratio."""
    p = prepare(returns, periods_per_year=1)
    m = _mar(mar, p.returns.index)

    def fn(s: pd.Series) -> float:
        d = np.maximum(m.values - s.values, 0)
        return float((s.mean() - m.mean()) / np.mean(d**order) ** (1 / order))
    return result(p, fn)


def upside_potential_ratio(returns: Any, mar: Any = 0.0, method: str = "subset") -> Any:
    """Sortino & van der Meer ``(sum(max(r-mar,0))/len) / downside_deviation(mar, method)``
    where ``len`` follows ``method``."""
    p = prepare(returns, periods_per_year=1)
    m = _mar(mar, p.returns.index)

    def fn(s: pd.Series) -> float:
        up = np.maximum(s.values - m.values, 0)
        n = len(up) if method == "full" else int((up > 0).sum())
        return float(up.sum() / n / _downside(s, m, method))
    return result(p, fn)


# --------------------------------------------------------------------------- gain/loss ratios
def d_ratio(returns: Any) -> Any:
    """``(n_down * sum|losses|) / (n_up * sum gains)`` (d ratio)."""
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        up, dn = s[s > 0], s[s < 0]
        return float(-len(dn) * dn.sum() / (len(up) * up.sum()))
    return result(p, fn)


def profit_factor(returns: Any) -> Any:
    """``sum(gains) / sum|losses|`` (profit factor / Bernardo-Ledoit ratio)."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(s[s > 0].sum() / -s[s < 0].sum()))


bernardo_ledoit_ratio = profit_factor


def gain_to_pain_ratio(returns: Any, freq: str | None = None) -> Any:
    """``sum(r) / sum|losses|`` (Jack Schwager's gain-to-pain ratio). ``freq`` sums returns to a lower
    frequency first (e.g. ``"ME"``)."""
    p = prepare(returns, periods_per_year=1)
    df = p.returns.resample(freq).sum() if freq else p.returns
    return result(p, lambda s: float(s.sum() / -s[s < 0].sum()), df)


def payoff_ratio(returns: Any) -> Any:
    """``mean(gains) / |mean(losses)|`` (payoff or win/loss ratio)."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(s[s > 0].mean() / -s[s < 0].mean()))


win_loss_ratio = payoff_ratio


def cpc_index(returns: Any) -> Any:
    """``profit_factor * win_rate * payoff_ratio`` (CPC index)."""
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        nz = s[s != 0]
        wr = (nz > 0).sum() / len(nz)
        return float(s[s > 0].sum() / -s[s < 0].sum() * wr * (s[s > 0].mean() / -s[s < 0].mean()))
    return result(p, fn)


def common_sense_ratio(returns: Any) -> Any:
    """``profit_factor * tail_ratio`` (common-sense ratio)."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(s[s > 0].sum() / -s[s < 0].sum()
                                     * abs(s.quantile(0.95) / s.quantile(0.05))))


def risk_of_ruin(returns: Any) -> Any:
    """``((1 - w) / (1 + w)) ** n`` with ``w`` the win rate (risk of ruin)."""
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        nz = s[s != 0]
        w = (nz > 0).sum() / len(nz)
        return float(((1 - w) / (1 + w)) ** len(s))
    return result(p, fn)


def kelly_ratio(returns: Any, rf: Any = 0.0, periods_per_year: int | None = None, half: bool = False,
                compounding: bool = True) -> Any:
    """Continuous Kelly leverage ``mean(r - rf) / var(r)``; ``half`` halves it."""
    p = prepare(returns, rf=rf, periods_per_year=periods_per_year, compounding=compounding)
    ex = p.excess

    def fn(s: pd.Series) -> float:
        k = ex[s.name].mean() / s.var(ddof=1)
        return float(k / 2 if half else k)
    return result(p, fn)


def kelly_criterion(returns: Any) -> Any:
    """Discrete Kelly fraction ``(payoff * w - (1 - w)) / payoff``."""
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        nz = s[s != 0]
        w = (nz > 0).sum() / len(nz)
        payoff = s[s > 0].mean() / -s[s < 0].mean()
        return float((payoff * w - (1 - w)) / payoff)
    return result(p, fn)


# --------------------------------------------------------------------------- behavioural / misc
def prospect_ratio(returns: Any, mar: float = 0.0) -> Any:
    """``(mean(max(r,0) + 2.25*min(r,0)) - mar) / downside_deviation(mar)`` (Watanabe 2006)."""
    p = prepare(returns, periods_per_year=1)
    m = _mar(mar, p.returns.index)

    def fn(s: pd.Series) -> float:
        v = s.values
        return float((np.mean(np.maximum(v, 0) + 2.25 * np.minimum(v, 0)) - mar) / _downside(s, m, "full"))
    return result(p, fn)


def skewness_kurtosis_ratio(returns: Any) -> Any:
    """Population skewness / population (non-excess) kurtosis."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(_skew(s) / _kurt(s)))


def volatility_skewness(returns: Any, mar: Any = 0.0, stat: str = "volatility") -> Any:
    """``upside variance / downside variance`` (``"volatility"``) or ``upside risk / downside
    deviation`` (``"variability"``), full-sample denominators."""
    p = prepare(returns, periods_per_year=1)
    m = _mar(mar, p.returns.index)

    def fn(s: pd.Series) -> float:
        up = np.maximum(s.values - m.values, 0)
        dd = _downside(s, m, "full")
        if stat == "volatility":
            return float(np.mean(up**2) / dd**2)
        if stat == "variability":
            return float(np.sqrt(np.mean(up**2)) / dd)
        raise ValueError("stat must be 'volatility' or 'variability'")
    return result(p, fn)


def risk_return_ratio(returns: Any) -> Any:
    """Un-annualised ``mean / std`` with no risk-free rate."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(s.mean() / s.std(ddof=1)))


def serenity_index(returns: Any, rf: Any = 0.0, ddof: int = 0) -> Any:
    """Serenity index ``(sum(r) - rf) / (ulcer_index * pitfall)`` where pitfall is
    ``-CVaR(drawdown series) / std(r)`` with CVaR = mean of drawdowns below the gaussian 95% VaR.
    ``ddof=1`` divides the ulcer index by ``n-1``. ``rf`` is subtracted as an annual scalar; a
    per-period Series is annualised geometrically first."""
    p = prepare(returns, periods_per_year=1)
    rf = rf_annual(rf, p.returns.index, None)

    def fn(s: pd.Series) -> float:
        wealth = (1 + s).cumprod()
        dd = wealth / np.maximum.accumulate(np.r_[1.0, wealth.values])[1:] - 1
        ulcer = np.sqrt((dd**2).sum() / (len(dd) - ddof))
        v = _var_series(dd, 0.95, "gaussian", "linear", 1)
        tail = dd[dd < v]
        cvar = tail.mean() if len(tail) else v
        pitfall = -cvar / s.std(ddof=1)
        return float((s.sum() - rf) / (ulcer * pitfall))
    return result(p, fn)
