"""Benchmark-relative statistics: CAPM regression, tracking, capture and selection measures.

Conventions:
* Regressions are OLS of excess fund returns on excess benchmark returns (``r - rf``).
* ``alpha`` is annualised by default (``(1+a)^ppy - 1``); ``annualize=False`` returns the
  per-period intercept, ``geometric=False`` annualises arithmetically (``a * ppy``).
* Bull/bear betas condition on the sign of the benchmark EXCESS return.
* Capture ratios compound the subset returns; the down-market subset is ``benchmark <= 0``.
  ``annualize=True`` gives the Morningstar variant (ratio of annualised subset returns).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from paris._core import Prepared, annualize_return, ols, pearson, pop_sd, prepare, safe_div

__all__ = [
    "active_premium",
    "alpha",
    "appraisal_ratio",
    "batting_average",
    "bear_beta",
    "beta",
    "bull_beta",
    "capture_ratio",
    "correlation",
    "down_capture",
    "down_number_ratio",
    "down_percentage_ratio",
    "fama_beta",
    "information_ratio",
    "jensen_alpha",
    "m_squared",
    "m_squared_excess",
    "modigliani",
    "net_selectivity",
    "r_squared",
    "regression_stats",
    "risk_premium",
    "selectivity",
    "specific_risk",
    "systematic_risk",
    "timing_ratio",
    "total_risk",
    "tracking_error",
    "treynor_ratio",
    "up_capture",
    "up_number_ratio",
    "up_percentage_ratio",
]


def _prep(returns: Any, benchmark: Any, rf: Any, ppy: int | None, compounding: bool = True,
          need_ppy: bool = True) -> Prepared:
    """Align inputs; frequency is only inferred when a metric needs it (``need_ppy``) or when a
    non-zero scalar rf must be de-annualised."""
    if benchmark is None:
        raise ValueError("benchmark is required for relative statistics")
    if ppy is None and not need_ppy and (not np.isscalar(rf) or rf == 0):
        ppy = 1
    return prepare(returns, benchmark=benchmark, rf=rf, periods_per_year=ppy, compounding=compounding)


def _each(p: Prepared, fn) -> Any:
    """Apply ``fn(excess_fund, excess_bench, fund_raw)`` per fund column."""
    xb = p.bench_excess
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.Series({c: fn(p.excess[c], xb, p.returns[c]) for c in p.returns.columns}, dtype=float)
    return out if p.multi else out.iloc[0]


# --------------------------------------------------------------------------- CAPM regression
def beta(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """CAPM beta: OLS slope of excess fund on excess benchmark returns."""
    p = _prep(returns, benchmark, rf, periods_per_year, need_ppy=False)
    return _each(p, lambda xa, xb, _: ols(xa.values, xb.values)[0])


def alpha(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None,
          annualize: bool = True, geometric: bool = True) -> Any:
    """CAPM alpha (regression intercept), annualised as ``(1+a)^ppy-1`` (or ``a*ppy``).
    ``annualize=False`` returns the per-period intercept."""
    p = _prep(returns, benchmark, rf, periods_per_year)

    def fn(xa, xb, _):
        a = ols(xa.values, xb.values)[1]
        if not annualize:
            return a
        return (1 + a) ** p.ppy - 1 if geometric else a * p.ppy
    return _each(p, fn)


def _cond_beta(p: Prepared, up: bool) -> Any:
    def fn(xa, xb, _):
        m = xb > 0 if up else xb < 0
        return ols(xa[m].values, xb[m].values)[0]
    return _each(p, fn)


def bull_beta(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """Beta over periods where the benchmark excess return is positive."""
    return _cond_beta(_prep(returns, benchmark, rf, periods_per_year, need_ppy=False), True)


def bear_beta(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """Beta over periods where the benchmark excess return is negative."""
    return _cond_beta(_prep(returns, benchmark, rf, periods_per_year, need_ppy=False), False)


def timing_ratio(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """``bull_beta / bear_beta`` (timing ratio)."""
    p = _prep(returns, benchmark, rf, periods_per_year, need_ppy=False)
    return _cond_beta(p, True) / _cond_beta(p, False)


def correlation(returns: Any, benchmark: Any) -> Any:
    """Pearson correlation of raw fund and benchmark returns."""
    p = _prep(returns, benchmark, 0.0, None, need_ppy=False)
    return _each(p, lambda xa, xb, _: pearson(xa.values, xb.values))


def r_squared(returns: Any, benchmark: Any, rf: Any = 0.0) -> Any:
    """R² of the CAPM regression on excess returns."""
    p = _prep(returns, benchmark, rf, None, need_ppy=False)
    return _each(p, lambda xa, xb, _: pearson(xa.values, xb.values) ** 2)


def regression_stats(returns: Any, benchmark: Any, rf: Any = 0.0,
                     periods_per_year: int | None = None) -> pd.DataFrame:
    """Per-fund CAPM regression table: alpha (per period), beta, r2, alpha_t, beta_t, alpha_p, beta_p
    (p-values need scipy; NaN otherwise), resid_sd (sample), n."""
    p = _prep(returns, benchmark, rf, periods_per_year)
    try:
        from scipy.stats import t as _t
    except ImportError:  # pragma: no cover - exercised only without scipy
        _t = None
    rows = {}
    xb = p.bench_excess.values
    keys = ["alpha", "beta", "r2", "alpha_t", "beta_t", "alpha_p", "beta_p", "resid_sd", "n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in p.returns.columns:
            y = p.excess[c].values
            n = len(y)
            if n < 3:  # no residual degrees of freedom: inference undefined
                rows[c] = dict.fromkeys(keys, np.nan) | {"n": n}
                continue
            b, a = ols(y, xb)
            resid = y - a - b * xb
            s2 = np.sum(resid**2) / (n - 2)
            sxx = np.sum((xb - xb.mean()) ** 2)
            se_b = np.sqrt(s2 / sxx)
            se_a = np.sqrt(s2 * (1 / n + xb.mean() ** 2 / sxx))
            ta, tb = a / se_a, b / se_b
            pa = pb = np.nan
            if _t is not None:
                pa, pb = 2 * _t.sf(abs(ta), n - 2), 2 * _t.sf(abs(tb), n - 2)
            rows[c] = {"alpha": a, "beta": b, "r2": pearson(y, xb) ** 2, "alpha_t": ta,
                       "beta_t": tb, "alpha_p": pa, "beta_p": pb, "resid_sd": float(np.sqrt(s2)), "n": n}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- Jensen / Treynor
def jensen_alpha(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """Annualised Jensen's alpha ``Rp - Rf - beta*(Rb - Rf)`` with geometric annualised returns."""
    p = _prep(returns, benchmark, rf, periods_per_year)
    rb = annualize_return(p.benchmark, p.ppy)
    rfa = annualize_return(p.rf, p.ppy)

    def fn(xa, xb, ra):
        b = ols(xa.values, xb.values)[0]
        return annualize_return(ra, p.ppy) - rfa - b * (rb - rfa)
    return _each(p, fn)


selectivity = jensen_alpha


def treynor_ratio(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None,
                  modified: bool = False) -> Any:
    """Annualised (geometric) excess return / beta (Treynor ratio); ``modified`` divides by
    systematic risk instead."""
    p = _prep(returns, benchmark, rf, periods_per_year)
    sdb = p.bench_excess.std(ddof=1) * np.sqrt(p.ppy)

    def fn(xa, xb, _):
        b = ols(xa.values, xb.values)[0]
        return safe_div(annualize_return(xa, p.ppy), b * sdb if modified else b)  # beta 0: inf / NaN
    return _each(p, fn)


# --------------------------------------------------------------------------- tracking
def tracking_error(returns: Any, benchmark: Any, periods_per_year: int | None = None) -> Any:
    """Annualised standard deviation of active returns (tracking error)."""
    p = _prep(returns, benchmark, 0.0, periods_per_year)
    return _each(p, lambda xa, xb, _: float((xa - xb).std(ddof=1) * np.sqrt(p.ppy)))


def active_premium(returns: Any, benchmark: Any, periods_per_year: int | None = None,
                   geometric: bool = True) -> Any:
    """Annualised fund return minus annualised benchmark return (active premium)."""
    p = _prep(returns, benchmark, 0.0, periods_per_year)
    rb = annualize_return(p.benchmark, p.ppy, geometric)
    return _each(p, lambda xa, xb, ra: annualize_return(ra, p.ppy, geometric) - rb)


def information_ratio(returns: Any, benchmark: Any, periods_per_year: int | None = None,
                      geometric: bool = True) -> Any:
    """``active_premium / tracking_error`` (information ratio). ``geometric=False`` uses the
    arithmetic annualised active return, i.e. ``mean(active)*ppy / (sd(active)*sqrt(ppy))``."""
    p = _prep(returns, benchmark, 0.0, periods_per_year)
    rb = annualize_return(p.benchmark, p.ppy, geometric)

    def fn(xa, xb, ra):
        te = (xa - xb).std(ddof=1) * np.sqrt(p.ppy)
        return (annualize_return(ra, p.ppy, geometric) - rb) / te
    return _each(p, fn)


# --------------------------------------------------------------------------- risk decomposition
def systematic_risk(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """``beta * annualised sd(excess benchmark)`` (systematic risk)."""
    p = _prep(returns, benchmark, rf, periods_per_year)
    sdb = p.bench_excess.std(ddof=1) * np.sqrt(p.ppy)
    return _each(p, lambda xa, xb, _: ols(xa.values, xb.values)[0] * sdb)


def _specific(xa: pd.Series, xb: pd.Series, ppy: int) -> float:
    b, a = ols(xa.values, xb.values)
    eps = xa.values - a - b * xb.values
    return float(np.sqrt(np.mean((eps - eps.mean()) ** 2)) * np.sqrt(ppy))


def specific_risk(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """Annualised population sd of CAPM residuals (specific risk)."""
    p = _prep(returns, benchmark, rf, periods_per_year)
    return _each(p, lambda xa, xb, _: _specific(xa, xb, p.ppy))


def total_risk(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """``sqrt(systematic_risk^2 + specific_risk^2)`` (total risk)."""
    p = _prep(returns, benchmark, rf, periods_per_year)
    sdb = p.bench_excess.std(ddof=1) * np.sqrt(p.ppy)
    return _each(p, lambda xa, xb, _: float(np.sqrt((ols(xa.values, xb.values)[0] * sdb) ** 2
                                                    + _specific(xa, xb, p.ppy) ** 2)))


def appraisal_ratio(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None,
                    method: str = "appraisal") -> Any:
    """Jensen's alpha / specific risk (``"appraisal"``), / beta (``"modified"``) or / systematic risk
    (``"alternative"``)."""
    p = _prep(returns, benchmark, rf, periods_per_year)
    rb, rfa = annualize_return(p.benchmark, p.ppy), annualize_return(p.rf, p.ppy)
    sdb = p.bench_excess.std(ddof=1) * np.sqrt(p.ppy)

    def fn(xa, xb, ra):
        b = ols(xa.values, xb.values)[0]
        ja = annualize_return(ra, p.ppy) - rfa - b * (rb - rfa)
        if method == "appraisal":
            return ja / _specific(xa, xb, p.ppy)
        if method == "modified":
            return ja / b
        if method == "alternative":
            return ja / (b * sdb)
        raise ValueError("method must be 'appraisal', 'modified' or 'alternative'")
    return _each(p, fn)


def fama_beta(returns: Any, benchmark: Any, periods_per_year: int | None = None) -> Any:
    """Ratio of annualised population sds, fund / benchmark (Fama beta)."""
    p = _prep(returns, benchmark, 0.0, periods_per_year)
    sdb = pop_sd(p.benchmark)
    return _each(p, lambda xa, xb, ra: safe_div(ra.std(ddof=0), sdb))  # sd / 0 = Inf, as R


def net_selectivity(returns: Any, benchmark: Any, rf: float = 0.0, periods_per_year: int | None = None) -> Any:
    """Fama's net selectivity: Jensen's alpha minus diversification cost
    ``(fama_beta - beta) * (Rb_ann - rf)`` (scalar annual ``rf``)."""
    p = _prep(returns, benchmark, rf, periods_per_year)
    rb, rfa = annualize_return(p.benchmark, p.ppy), annualize_return(p.rf, p.ppy)
    sdb = pop_sd(p.benchmark)

    def fn(xa, xb, ra):
        b = ols(xa.values, xb.values)[0]
        ja = annualize_return(ra, p.ppy) - rfa - b * (rb - rfa)
        return ja - (safe_div(ra.std(ddof=0), sdb) - b) * (rb - rfa)
    return _each(p, fn)


# --------------------------------------------------------------------------- M-squared
def m_squared(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """Modigliani-Modigliani M²: ``(Rp - rf) * sigma_b / sigma_p + rf`` with geometric annualised
    return and annualised POPULATION sds. A per-period rf Series is annualised
    geometrically."""
    p = _prep(returns, benchmark, rf, periods_per_year)
    rfa = annualize_return(p.rf, p.ppy)
    sigm = p.benchmark.std(ddof=0) * np.sqrt(p.ppy)
    return _each(p, lambda xa, xb, ra: (annualize_return(ra, p.ppy) - rfa) * sigm
                 / (ra.std(ddof=0) * np.sqrt(p.ppy)) + rfa)


def m_squared_excess(returns: Any, benchmark: Any, rf: float = 0.0, periods_per_year: int | None = None,
                     geometric: bool = True) -> Any:
    """M² minus the annualised benchmark return, geometric ``(1+M2)/(1+Rb)-1`` or arithmetic."""
    p = _prep(returns, benchmark, rf, periods_per_year)
    m2 = m_squared(returns, benchmark, rf, periods_per_year)
    rb = annualize_return(p.benchmark, p.ppy)
    return (1 + m2) / (1 + rb) - 1 if geometric else m2 - rb


def modigliani(returns: Any, benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """Per-period Modigliani measure ``sharpe_period * sd(benchmark) + mean(rf)``."""
    p = _prep(returns, benchmark, rf, periods_per_year, need_ppy=False)
    sdb, mrf = p.benchmark.std(ddof=1), p.rf.mean()
    return _each(p, lambda xa, xb, _: float(xa.mean() / xa.std(ddof=1) * sdb + mrf))


# --------------------------------------------------------------------------- capture / batting
def _capture(p: Prepared, up: bool, geometric: bool, annualize: bool) -> Any:
    def fn(xa, xb, ra):
        m = p.benchmark > 0 if up else p.benchmark <= 0
        a, b = ra[m], p.benchmark[m]
        if not geometric:
            return float(a.sum() / b.sum())
        if annualize:
            return annualize_return(a, p.ppy) / annualize_return(b, p.ppy)
        return float((np.prod(1 + a.values) - 1) / (np.prod(1 + b.values) - 1))
    return _each(p, fn)


def up_capture(returns: Any, benchmark: Any, periods_per_year: int | None = None,
               geometric: bool = True, annualize: bool = False) -> Any:
    """Compound fund return / compound benchmark return over periods with benchmark > 0;
    ``annualize=True`` gives the Morningstar variant."""
    return _capture(prepare(returns, benchmark, periods_per_year=periods_per_year if annualize else 1),
                    True, geometric, annualize)


def down_capture(returns: Any, benchmark: Any, periods_per_year: int | None = None,
                 geometric: bool = True, annualize: bool = False) -> Any:
    """As :func:`up_capture` over periods with benchmark <= 0."""
    return _capture(prepare(returns, benchmark, periods_per_year=periods_per_year if annualize else 1),
                    False, geometric, annualize)


def capture_ratio(returns: Any, benchmark: Any, periods_per_year: int | None = None,
                  geometric: bool = True, annualize: bool = False) -> Any:
    """``up_capture / down_capture``."""
    return (up_capture(returns, benchmark, periods_per_year, geometric, annualize)
            / down_capture(returns, benchmark, periods_per_year, geometric, annualize))


def up_number_ratio(returns: Any, benchmark: Any) -> Any:
    """Share of benchmark-up periods in which the fund was also up (up number ratio)."""
    p = prepare(returns, benchmark, periods_per_year=1)
    return _each(p, lambda xa, xb, ra: float(((ra > 0) & (p.benchmark > 0)).sum() / (p.benchmark > 0).sum()))


def down_number_ratio(returns: Any, benchmark: Any) -> Any:
    """Share of benchmark-down periods in which the fund was also down."""
    p = prepare(returns, benchmark, periods_per_year=1)
    return _each(p, lambda xa, xb, ra: float(((ra < 0) & (p.benchmark < 0)).sum() / (p.benchmark < 0).sum()))


def up_percentage_ratio(returns: Any, benchmark: Any) -> Any:
    """Share of benchmark-up periods in which the fund beat the benchmark (up percentage ratio)."""
    p = prepare(returns, benchmark, periods_per_year=1)
    return _each(p, lambda xa, xb, ra: float(((ra > p.benchmark) & (p.benchmark > 0)).sum()
                                             / (p.benchmark > 0).sum()))


def down_percentage_ratio(returns: Any, benchmark: Any) -> Any:
    """Share of benchmark-down periods in which the fund beat the benchmark (down percentage ratio)."""
    p = prepare(returns, benchmark, periods_per_year=1)
    return _each(p, lambda xa, xb, ra: float(((ra > p.benchmark) & (p.benchmark < 0)).sum()
                                             / (p.benchmark < 0).sum()))


def batting_average(returns: Any, benchmark: Any) -> Any:
    """Share of all periods in which the fund beat the benchmark (batting average)."""
    p = prepare(returns, benchmark, periods_per_year=1)
    return _each(p, lambda xa, xb, ra: float((ra > p.benchmark).mean()))


def risk_premium(benchmark: Any, rf: Any = 0.0, periods_per_year: int | None = None) -> Any:
    """Mean per-period excess return of a series."""
    p = prepare(benchmark, rf=rf, periods_per_year=periods_per_year)
    out = p.excess.mean()
    return out if p.multi else float(out.iloc[0])
