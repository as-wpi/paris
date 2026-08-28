"""Dispersion, downside and tail-risk measures.

Conventions:
* ``volatility`` is the sample standard deviation (ddof=1), annualised by ``sqrt(ppy)``;
  ``gain_deviation`` / ``loss_deviation`` are the same estimator on the gains / losses only.
* ``downside_deviation(method="full")`` divides by the full sample size n; ``"subset"`` by the
  count of observations below MAR.
* ``skewness``/``kurtosis`` default to population moments (``"moment"`` / ``"excess"``); pandas'
  ``.skew()``/``.kurt()`` correspond to ``method="sample"`` / ``"sample_excess"``.
* ``var``/``cvar`` return a (normally negative) return quantile, i.e. loss expressed as a return.
  ``method="modified"`` is the Cornish-Fisher expansion.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from paris._core import (  # noqa: F401
    _cdf,
    _central,
    _downside,
    _es_series,
    _pdf,
    _ppf,
    _std_moment,
    _var_series,
    annualize_vol,
    prepare,
    result,
)

__all__ = [
    "cvar",
    "downside_deviation",
    "downside_frequency",
    "gain_deviation",
    "kurtosis",
    "loss_deviation",
    "mean_absolute_deviation",
    "outlier_loss_ratio",
    "outlier_win_ratio",
    "semi_deviation",
    "semi_variance",
    "skewness",
    "tail_ratio",
    "upside_frequency",
    "upside_risk",
    "var",
    "volatility",
]


# --------------------------------------------------------------------------- dispersion
def volatility(returns: Any, periods_per_year: int | None = None, annualize: bool = True,
               ddof: int = 1) -> Any:
    """Standard deviation of returns, annualised with ``sqrt(ppy)``."""
    p = prepare(returns, periods_per_year=periods_per_year if annualize else 1)
    return result(p, lambda s: annualize_vol(s.std(ddof=ddof), p.ppy if annualize else 1))


def _mar_series(mar: Any, index: pd.Index) -> pd.Series:
    """MAR is a PER-PERIOD threshold; a Series is aligned to the returns index."""
    if np.isscalar(mar):
        return pd.Series(float(mar), index=index)
    if isinstance(mar, pd.Series):
        return mar.reindex(index)
    return pd.Series(np.asarray(mar, dtype=float).ravel(), index=index)


def downside_deviation(returns: Any, mar: Any = 0.0, method: str = "full",
                       periods_per_year: int | None = None, annualize: bool = False) -> Any:
    """Downside deviation below a per-period MAR.

    Set ``annualize=True`` to scale by ``sqrt(ppy)`` (the annualised Sortino denominator).
    """
    p = prepare(returns, periods_per_year=periods_per_year if annualize else 1)
    m = _mar_series(mar, p.returns.index)
    return result(p, lambda s: _downside(s, m, method, 2, p.ppy if annualize else 1))


def upside_risk(returns: Any, mar: Any = 0.0, method: str = "full", stat: str = "risk") -> Any:
    """Upside counterpart of downside deviation: ``stat`` in risk/variance/potential."""
    p = prepare(returns, periods_per_year=1)
    m = _mar_series(mar, p.returns.index)

    def fn(s: pd.Series) -> float:
        u = np.maximum(s.values - m.values, 0.0)
        n = len(u) if method == "full" else int((s.values > m.values).sum())
        if method not in ("full", "subset"):
            raise ValueError("method must be 'full' or 'subset'")
        if n == 0:
            return float("nan")
        if stat == "risk":
            return float(math.sqrt(np.sum(u ** 2) / n))
        if stat == "variance":
            return float(np.sum(u ** 2) / n)
        if stat == "potential":
            return float(np.sum(u) / n)
        raise ValueError("stat must be 'risk', 'variance' or 'potential'")
    return result(p, fn)


def semi_deviation(returns: Any) -> Any:
    """Downside deviation below the sample mean, full-sample denominator."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: _downside(s, pd.Series(s.mean(), index=s.index), "full"))


def semi_variance(returns: Any) -> Any:
    """Squared downside deviation below the mean, SUBSET denominator.

    Note: not the square of :func:`semi_deviation` — the two use different denominators.
    """
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: _downside(s, pd.Series(s.mean(), index=s.index), "subset") ** 2)


def gain_deviation(returns: Any) -> Any:
    """Sample standard deviation of the strictly positive returns (gain deviation); NaN with fewer
    than two gains."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: _side_sd(s.values[s.values > 0]))


def loss_deviation(returns: Any) -> Any:
    """Sample standard deviation of the strictly negative returns (loss deviation); NaN with fewer
    than two losses."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: _side_sd(s.values[s.values < 0]))


def _side_sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) >= 2 else float("nan")


def mean_absolute_deviation(returns: Any) -> Any:
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(np.abs(s.values - s.mean()).mean()))


# --------------------------------------------------------------------------- moments
def skewness(returns: Any, method: str = "moment") -> Any:
    """Skewness: ``"moment"`` (population, default), ``"fisher"`` (== pandas ``.skew()``) or
    ``"sample"`` (population sigma scaled by ``n/((n-1)(n-2))``)."""
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        x, n = s.values, len(s)
        g = _std_moment(x, 3)  # NaN for a constant series (R: 0/0)
        if method == "moment":
            return g
        if method == "fisher":
            return float(math.sqrt(n * (n - 1)) / (n - 2) * g) if n > 2 else float("nan")
        if method == "sample":
            if n <= 2 or math.isnan(g):
                return float("nan")
            return float(np.sum((x - x.mean()) ** 3) / _central(x, 2) ** 1.5 * n / ((n - 1) * (n - 2)))
        raise ValueError("method must be 'moment', 'fisher' or 'sample'")
    return result(p, fn)


def kurtosis(returns: Any, method: str = "excess") -> Any:
    """Kurtosis: ``"excess"`` (population, default), ``"moment"``, ``"sample"`` or
    ``"sample_excess"`` (== pandas ``.kurt()``). ``"fisher"`` is an alias of ``"sample_excess"``
    (algebraically identical)."""
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        x, n = s.values, len(s)
        k = _std_moment(x, 4)  # NaN for a constant series (R: 0/0)
        if method == "moment":
            return k
        if method == "excess":
            return k - 3.0
        if method in ("sample", "sample_excess", "fisher"):
            if n <= 3 or math.isnan(k):
                return float("nan")
            m4 = np.sum((x - x.mean()) ** 4)
            ks = n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * m4 / s.std(ddof=1) ** 4
            return float(ks if method == "sample" else ks - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        raise ValueError("method must be 'excess', 'moment', 'fisher', 'sample' or 'sample_excess'")
    return result(p, fn)


# --------------------------------------------------------------------------- VaR / ES
def var(returns: Any, confidence: float = 0.95, method: str = "historical",
        interpolation: str = "linear", ddof: int = 0) -> Any:
    """Value at Risk as a return quantile (negative = loss).

    ``method``: ``"historical"`` (empirical quantile, ``interpolation`` as in ``numpy.quantile``;
    ``"linear"`` is Hyndman-Fan type 7), ``"gaussian"`` (``mu + z*sigma``) or ``"modified"``
    (Cornish-Fisher). Parametric methods use the population sigma (``ddof=0``); ``ddof=1`` uses the
    sample sigma. Positive (inverse-risk) values are returned, not replaced by NaN.
    """
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: _var_series(s, confidence, method, interpolation, ddof))


def cvar(returns: Any, confidence: float = 0.95, method: str = "historical",
         interpolation: str = "linear", ddof: int = 0, operational: bool = True) -> Any:
    """Conditional VaR / Expected Shortfall: mean return in the tail beyond VaR.

    ``method="modified"`` is the Cornish-Fisher ES of Boudt, Peterson & Croux (2008);
    ``operational=True`` (default) caps it so that ES is never less severe than modified VaR.
    ``method="gaussian_tail"`` is the empirical mean of returns below the gaussian VaR.
    """
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: _es_series(s, confidence, method, interpolation, ddof, operational))


expected_shortfall = cvar


# --------------------------------------------------------------------------- tails
def tail_ratio(returns: Any, cutoff: float = 0.95) -> Any:
    """``|q(cutoff) / q(1-cutoff)|`` (tail ratio)."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(abs(s.quantile(cutoff) / s.quantile(1 - cutoff))))


def outlier_win_ratio(returns: Any, quantile: float = 0.99) -> Any:
    """``q(quantile) / mean(positive returns)`` (outlier win ratio)."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(s.quantile(quantile) / s[s > 0].mean()))


def outlier_loss_ratio(returns: Any, quantile: float = 0.01) -> Any:
    """``q(quantile) / mean(negative returns)`` (outlier loss ratio)."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(s.quantile(quantile) / s[s < 0].mean()))


def upside_frequency(returns: Any, mar: float = 0.0) -> Any:
    """Share of periods with return > MAR."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float((s > mar).mean()))


def downside_frequency(returns: Any, mar: float = 0.0) -> Any:
    """Share of periods with return < MAR."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float((s < mar).mean()))
