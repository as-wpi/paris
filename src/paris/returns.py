"""Return measures: cumulative/total, annualised, CAGR, period (MTD..ITD), calendar, win/loss stats.

Conventions:
* ``geometric=True`` compounds returns; ``False`` sums them.
* ``annualized_return`` scales by observation count: ``prod(1+r)^(ppy/n) - 1``.
* ``cagr(method="calendar")`` uses elapsed calendar days / ``days_in_year``; ``method="periods"``
  is period-count based and equals ``annualized_return``.
* ``win_rate`` excludes zero returns by default (``include_zeros=True`` keeps them).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from paris._core import AlignmentError, annualize_return, prepare, result, to_frame

__all__ = [
    "active_returns",
    "aggregate",
    "annualized_return",
    "avg_loss",
    "avg_return",
    "avg_win",
    "best",
    "cagr",
    "calendar_returns",
    "consecutive_losses",
    "consecutive_wins",
    "cumulative_returns",
    "excess_returns",
    "period_returns",
    "total_return",
    "wealth_index",
    "win_rate",
    "worst",
]

_TRAILING = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}
_WINDOWS = ["MTD", "QTD", "YTD", "1Y", "3Y", "5Y", "10Y", "ITD"]


def _unwrap(df: pd.DataFrame, multi: bool) -> Any:
    return df if multi else df.iloc[:, 0]


# --------------------------------------------------------------------------- cumulative
def cumulative_returns(returns: Any, geometric: bool = True) -> Any:
    """Cumulative return path: ``cumprod(1+r)-1`` (or ``cumsum`` if not geometric)."""
    p = prepare(returns, periods_per_year=1)
    out = (1 + p.returns).cumprod() - 1 if geometric else p.returns.cumsum()
    return _unwrap(out, p.multi)


def wealth_index(returns: Any, start: float = 1.0) -> Any:
    """Growth of ``start`` invested at the beginning of the series."""
    return (1 + cumulative_returns(returns)) * start


def total_return(returns: Any, geometric: bool = True) -> Any:
    """Total (cumulative) return over the full sample."""
    p = prepare(returns, periods_per_year=1)
    fn = (lambda s: float(np.prod(1 + s.values) - 1)) if geometric else (lambda s: float(s.sum()))
    return result(p, fn)


def annualized_return(returns: Any, periods_per_year: int | None = None, geometric: bool = True) -> Any:
    """Annualised return, ``prod(1+r)^(ppy/n)-1``."""
    p = prepare(returns, periods_per_year=periods_per_year)
    return result(p, lambda s: annualize_return(s, p.ppy, geometric))


def cagr(
    returns: Any,
    periods_per_year: int | None = None,
    method: str = "periods",
    days_in_year: float = 365.25,
    start: Any = None,
) -> Any:
    """Compound annual growth rate.

    ``method="periods"``: identical to :func:`annualized_return`.
    ``method="calendar"``: ``(1+total)^(days_in_year/elapsed_days)-1`` with elapsed days measured
    from ``start`` (the date of the initial value, e.g. the first price date) or, by default, from
    the first return date. Requires a DatetimeIndex.
    """
    if method == "periods":
        return annualized_return(returns, periods_per_year)
    if method != "calendar":
        raise ValueError("method must be 'periods' or 'calendar'")
    p = prepare(returns, periods_per_year=1)
    first = pd.Timestamp(start) if start is not None else p.returns.index[0]
    days = (p.returns.index[-1] - first).days
    return result(p, lambda s: float(np.prod(1 + s.values) ** (days_in_year / days) - 1))


# --------------------------------------------------------------------------- aggregation
def aggregate(returns: Any, freq: str = "ME", geometric: bool = True) -> Any:
    """Compound (or sum) periodic returns to a lower frequency, e.g. ``"ME"``, ``"QE"``, ``"YE"``."""
    p = prepare(returns, periods_per_year=1)
    g = p.returns.resample(freq)
    out = g.apply(lambda s: np.prod(1 + s.values) - 1) if geometric else g.sum()
    return _unwrap(out, p.multi)


def calendar_returns(returns: Any, freq: str = "YE", geometric: bool = True) -> pd.DataFrame:
    """Table of returns per calendar period (rows) by fund (columns); yearly by default."""
    out = to_frame(aggregate(returns, freq, geometric))
    if freq.upper().startswith("Y"):
        out.index = out.index.year
    return out


def best(returns: Any, freq: str | None = None) -> Any:
    """Best periodic return, optionally after aggregating to ``freq`` (e.g. best month from daily)."""
    r = aggregate(returns, freq) if freq else returns
    return result(prepare(r, periods_per_year=1), lambda s: float(s.max()))


def worst(returns: Any, freq: str | None = None) -> Any:
    r = aggregate(returns, freq) if freq else returns
    return result(prepare(r, periods_per_year=1), lambda s: float(s.min()))


# --------------------------------------------------------------------------- win / loss
def win_rate(returns: Any, include_zeros: bool = False) -> Any:
    """Share of periods with positive return (zero returns excluded by default)."""
    def fn(s: pd.Series) -> float:
        d = s if include_zeros else s[s != 0]
        return float((d > 0).sum() / len(d)) if len(d) else float("nan")
    return result(prepare(returns, periods_per_year=1), fn)


def avg_win(returns: Any) -> Any:
    """Mean of positive returns."""
    return result(prepare(returns, periods_per_year=1),
                  lambda s: float(s[s > 0].mean()) if (s > 0).any() else float("nan"))


def avg_loss(returns: Any) -> Any:
    """Mean of negative returns."""
    return result(prepare(returns, periods_per_year=1),
                  lambda s: float(s[s < 0].mean()) if (s < 0).any() else float("nan"))


def avg_return(returns: Any, include_zeros: bool = False) -> Any:
    """Mean periodic return (zero returns excluded by default)."""
    return result(prepare(returns, periods_per_year=1),
                  lambda s: float((s if include_zeros else s[s != 0]).mean()))


def _max_run(mask: np.ndarray) -> int:
    best_run = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best_run = max(best_run, run)
    return best_run


def consecutive_wins(returns: Any) -> Any:
    """Longest streak of positive returns."""
    return result(prepare(returns, periods_per_year=1), lambda s: _max_run(s.values > 0))


def consecutive_losses(returns: Any) -> Any:
    """Longest streak of negative returns."""
    return result(prepare(returns, periods_per_year=1), lambda s: _max_run(s.values < 0))


# --------------------------------------------------------------------------- excess / active
def excess_returns(
    returns: Any, rf: Any = 0.0, periods_per_year: int | None = None,
    compounding: bool = True, geometric: bool = False,
) -> Any:
    """Returns in excess of the risk-free rate.

    ``geometric=False`` (default): ``r - rf``;
    ``geometric=True``: ``(1+r)/(1+rf) - 1``. ``compounding`` controls scalar-rf de-annualisation.
    """
    p = prepare(returns, rf=rf, periods_per_year=periods_per_year, compounding=compounding)
    out = (1 + p.returns).div(1 + p.rf, axis=0) - 1 if geometric else p.excess
    return _unwrap(out, p.multi)


def active_returns(returns: Any, benchmark: Any) -> Any:
    """Arithmetic active return ``r - benchmark`` on the common window."""
    p = prepare(returns, benchmark=benchmark, periods_per_year=1)
    return _unwrap(p.active, p.multi)


# --------------------------------------------------------------------------- period returns
def period_returns(
    returns: Any,
    as_of: Any = None,
    windows: Sequence[str] = _WINDOWS,
    periods_per_year: int | None = None,
) -> pd.DataFrame:
    """Trailing and to-date returns (rows) per fund (columns), as of the last observation.

    MTD/QTD/YTD compound observations dated within the current calendar month/quarter/year.
    ``1Y/3Y/5Y/10Y`` use observations dated after ``as_of - N years`` and are annualised by
    exactly N years (NaN when history is insufficient). ITD is the full sample, annualised
    with ``annualized_return`` when it spans at least one year.
    """
    bad = [w for w in windows if w not in _WINDOWS]
    if bad:
        raise ValueError(f"unknown windows {bad}; choose from {_WINDOWS}")
    p = prepare(returns, periods_per_year=periods_per_year)
    df = p.returns
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AlignmentError("period_returns requires a DatetimeIndex")
    as_of = pd.Timestamp(as_of) if as_of is not None else df.index[-1]
    df = df.loc[:as_of]
    first = df.index[0]
    tol = pd.Timedelta(days=max(7.0, 1.2 * 366 / p.ppy))  # one period of slack for start alignment

    def tot(sub: pd.DataFrame) -> pd.Series:
        return (1 + sub).prod() - 1

    rows = {}
    for w in windows:
        if w == "MTD":
            rows[w] = tot(df.loc[as_of.to_period("M").start_time:])
        elif w == "QTD":
            rows[w] = tot(df.loc[as_of.to_period("Q").start_time:])
        elif w == "YTD":
            rows[w] = tot(df.loc[as_of.to_period("Y").start_time:])
        elif w == "ITD":
            n = len(df)
            rows[w] = tot(df) if n < p.ppy else (1 + tot(df)) ** (p.ppy / n) - 1
        else:
            years = _TRAILING[w]
            start = as_of - pd.DateOffset(years=years)
            if first > start + tol:
                rows[w] = pd.Series(np.nan, index=df.columns)
            else:
                rows[w] = (1 + tot(df.loc[df.index > start])) ** (1 / years) - 1
    return pd.DataFrame(rows).T.loc[list(windows)]
