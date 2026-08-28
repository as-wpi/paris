"""Tables: plain ``pd.DataFrame`` compositions of the scalar statistics, plus a rolling-window helper.

Every cell is produced by a public function of the topic modules with the conventions stated in
its own docstring — nothing is recomputed here, so a table is exactly as validated as the
functions it calls. Layout conventions shared by all tables:

* rows = metric labels (the strings of ``summary.ABSOLUTE_METRICS`` / ``RELATIVE_METRICS`` where the
  same statistic appears there) or periods; columns = funds, benchmark last when one is given;
* numbers are decimals (0.05 = 5 %), drawdowns negative, VaR/CVaR as (negative) returns;
* the signature order is ``returns, [benchmark], [rf], periods_per_year, *, switches`` and a
  DataFrame input yields one wide table, never a dict of tables.
* ``rolling`` windows are counted in observations; leading incomplete windows are **trimmed** by
  default (``trim=False`` pads them with NaN).
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from paris import drawdown as D
from paris import ratios as Q
from paris import returns as R
from paris import risk as K
from paris._core import FrequencyError, Prepared, prepare
from paris.summary import ABSOLUTE_METRICS as ABS
from paris.summary import RELATIVE_METRICS as REL

__all__ = [
    "annualized_table",
    "calendar_table",
    "capture_table",
    "distribution_table",
    "downside_table",
    "drawdown_ratio_table",
    "drawdown_summary",
    "rolling",
]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- helpers
def _columns(p: Prepared, with_benchmark: bool) -> dict[str, pd.Series]:
    """Fund columns in input order, benchmark last (named ``benchmark`` if unnamed)."""
    cols = {c: p.returns[c] for c in p.returns.columns}
    if with_benchmark and p.benchmark is not None:
        cols[p.benchmark.name or "benchmark"] = p.benchmark
    return cols


def _build(rows: dict[str, Callable[[pd.Series], Any]], cols: dict[str, pd.Series]) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({name: {label: fn(s) for label, fn in rows.items()} for name, s in cols.items()})
    out = out.reindex(list(rows))
    out.index.name = "metric"
    return out


def _abs(label: str, p: Prepared) -> Callable[[pd.Series], Any]:
    """The registry callable for *label* with this table's rf / periods-per-year bound."""
    fn = ABS[label]
    return lambda s: fn(s, rf=p.rf, ppy=p.ppy)


def _rel(label: str, p: Prepared) -> Callable[[pd.Series], Any]:
    fn = REL[label]
    return lambda s: fn(s, p.benchmark, p.rf, p.ppy)


# --------------------------------------------------------------------------- tables
def capture_table(returns: Any, benchmark: Any, periods_per_year: int | None = None) -> pd.DataFrame:
    """Up/down capture, number and percentage ratios per fund."""
    p = prepare(returns, benchmark=benchmark, periods_per_year=periods_per_year)
    labels = ["Up Capture", "Down Capture", "Up Number Ratio", "Down Number Ratio",
              "Up Percentage Ratio", "Down Percentage Ratio"]
    return _build({lab: _rel(lab, p) for lab in labels}, _columns(p, with_benchmark=False))


def downside_table(returns: Any, benchmark: Any = None, rf: Any = 0.0, periods_per_year: int | None = None,
                   *, mar: Any = None, confidence: float = 0.95) -> pd.DataFrame:
    """Downside-risk table: semi/gain/loss deviation, downside deviation
    below ``mar`` (default 10 % a year, i.e. ``0.10 / periods_per_year`` per period), below the
    per-period ``rf`` and below 0, maximum drawdown, and historical / modified VaR and CVaR at
    ``confidence``."""
    p = prepare(returns, benchmark=benchmark, rf=rf, periods_per_year=periods_per_year)
    mar_pp = 0.10 / p.ppy if mar is None else mar
    pct = f"{confidence:.0%}"
    rows: dict[str, Callable[[pd.Series], Any]] = {
        "Semi Deviation": _abs("Semi Deviation", p),
        "Gain Deviation": _abs("Gain Deviation", p),
        "Loss Deviation": _abs("Loss Deviation", p),
        "Downside Deviation (MAR)": lambda s: K.downside_deviation(s, mar=mar_pp),
        "Downside Deviation (rf)": lambda s: K.downside_deviation(s, mar=p.rf),
        "Downside Deviation (0)": lambda s: K.downside_deviation(s),
        "Max Drawdown": _abs("Max Drawdown", p),
        f"VaR {pct} (hist.)": lambda s: K.var(s, confidence),
        f"CVaR {pct} (hist.)": lambda s: K.cvar(s, confidence),
        f"VaR {pct} (modified)": lambda s: K.var(s, confidence, method="modified"),
        f"CVaR {pct} (modified)": lambda s: K.cvar(s, confidence, method="modified"),
    }
    return _build(rows, _columns(p, with_benchmark=True))


def distribution_table(returns: Any, benchmark: Any = None, periods_per_year: int | None = None) -> pd.DataFrame:
    """Per-period volatility and the five moment estimators."""
    p = prepare(returns, benchmark=benchmark, periods_per_year=periods_per_year)
    rows = {
        "Volatility (period)": lambda s: K.volatility(s, annualize=False),
        "Skewness": _abs("Skewness", p),
        "Kurtosis": lambda s: K.kurtosis(s, method="moment"),
        "Kurtosis (excess)": _abs("Kurtosis (excess)", p),
        "Skewness (sample)": lambda s: K.skewness(s, method="sample"),
        "Kurtosis (sample excess)": lambda s: K.kurtosis(s, method="sample_excess"),
    }
    return _build(rows, _columns(p, with_benchmark=True))


def annualized_table(returns: Any, benchmark: Any = None, rf: Any = 0.0, periods_per_year: int | None = None,
                     *, geometric: bool = True) -> pd.DataFrame:
    """Annualised return, volatility and Sharpe ratio; ``geometric``
    switches both the return and the Sharpe numerator to arithmetic."""
    p = prepare(returns, benchmark=benchmark, rf=rf, periods_per_year=periods_per_year)
    rows = {
        "CAGR": lambda s: R.annualized_return(s, p.ppy, geometric=geometric),
        "Volatility (ann.)": _abs("Volatility (ann.)", p),
        "Sharpe": lambda s: Q.sharpe(s, p.rf, p.ppy, geometric=geometric),
    }
    return _build(rows, _columns(p, with_benchmark=True))


def calendar_table(returns: Any, benchmark: Any = None, periods_per_year: int | None = None,
                   *, geometric: bool = True) -> pd.DataFrame:
    """Calendar grid of monthly returns (rows = years, columns = Jan..Dec + ``Annual``).
    Daily or weekly input is first compounded to months
    (:func:`paris.returns.aggregate`); coarser input raises ``FrequencyError``. A Series gives the
    flat grid plus the benchmark's annual return as a last column; a DataFrame gives one grid per
    fund under MultiIndex columns ``(fund, month)`` with the benchmark's grid last."""
    p = prepare(returns, benchmark=benchmark, periods_per_year=periods_per_year)
    if p.ppy < 12:
        raise FrequencyError("calendar_table needs monthly or higher-frequency returns")
    cols = _columns(p, with_benchmark=True)
    grids = {}
    for name, s in cols.items():
        monthly = R.aggregate(s, "ME", geometric)
        annual = R.aggregate(s, "YE", geometric)
        g = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "r": monthly.values})
        g = g.pivot(index="year", columns="month", values="r").reindex(columns=range(1, 13))
        g.columns = MONTHS
        g["Annual"] = pd.Series(annual.values, index=annual.index.year)
        grids[name] = g
    if not p.multi:
        fund = p.returns.columns[0]
        out = grids[fund]
        if p.benchmark is not None:
            bname = p.benchmark.name or "benchmark"
            out[bname] = grids[bname]["Annual"]
    else:
        out = pd.concat(grids, axis=1)
    out.index.name = "year"
    return out


def drawdown_summary(returns: Any, benchmark: Any = None, periods_per_year: int | None = None) -> pd.DataFrame:
    """Depth, length and recovery statistics of the drawdown episodes, plus the ulcer and pain
    indices, per fund (each row is the scalar function of the same name)."""
    p = prepare(returns, benchmark=benchmark, periods_per_year=periods_per_year)
    labels = ["Max Drawdown", "Average Drawdown", "Longest Drawdown (periods)", "Average Length (periods)",
              "Average Recovery (periods)", "Current Drawdown", "Ulcer Index", "Pain Index"]
    return _build({lab: _abs(lab, p) for lab in labels}, _columns(p, with_benchmark=True))


def drawdown_ratio_table(returns: Any, benchmark: Any = None, rf: Any = 0.0,
                         periods_per_year: int | None = None) -> pd.DataFrame:
    """Drawdown-based reward/risk ratios: Sterling, Calmar, Burke, Pain
    and Martin. Unlike the registry rows of ``stats()`` (rf = 0), ``rf`` is applied to the three
    ratios that take one."""
    p = prepare(returns, benchmark=benchmark, rf=rf, periods_per_year=periods_per_year)
    rows = {
        "Sterling": lambda s: D.sterling_ratio(s, periods_per_year=p.ppy),
        "Calmar": lambda s: D.calmar_ratio(s, p.ppy),
        "Burke": lambda s: D.burke_ratio(s, rf=p.rf, periods_per_year=p.ppy),
        "Pain Ratio": lambda s: D.pain_ratio(s, rf=p.rf, periods_per_year=p.ppy),
        "Martin": lambda s: D.martin_ratio(s, rf=p.rf, periods_per_year=p.ppy),
    }
    return _build(rows, _columns(p, with_benchmark=True))


# --------------------------------------------------------------------------- rolling
def rolling(returns: Any, fn: Callable[..., Any], window: int, benchmark: Any = None, rf: Any = 0.0,
            periods_per_year: int | None = None, *, trim: bool = True, **kwargs: Any) -> Any:
    """Apply a scalar PARIS function over a sliding window of ``window`` observations.

    ``fn`` is called on each window exactly as a user would call it — with the benchmark and rf
    slices if its signature takes ``benchmark`` / ``rf``, the resolved ``periods_per_year``, and
    ``**kwargs`` — so every value is the tested scalar. Returns a Series (Series input) or a
    DataFrame (one column per fund) indexed by the window's last date; rows before the first
    complete window are dropped (``trim=True``) or NaN (``trim=False``).
    """
    p = prepare(returns, benchmark=benchmark, rf=rf, periods_per_year=periods_per_year)
    n = len(p.returns)
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must be an integer >= 2")
    if window > n:
        raise ValueError(f"window ({window}) exceeds the {n} observations of the common window")
    params = inspect.signature(fn).parameters
    ppy = {"periods_per_year": p.ppy} if "periods_per_year" in params else {}
    ends = range(window - 1, n)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for col in p.returns.columns:
            s = p.returns[col]
            vals = []
            for i in ends:
                lo, hi = i - window + 1, i + 1
                args: list[Any] = [s.iloc[lo:hi]]
                kw: dict[str, Any] = dict(ppy, **kwargs)
                if "benchmark" in params:
                    args.append(p.benchmark.iloc[lo:hi])
                if "rf" in params:
                    kw["rf"] = p.rf.iloc[lo:hi]
                vals.append(fn(*args, **kw))
            out[col] = vals
    res = pd.DataFrame(out, index=p.returns.index[window - 1:], dtype=float)
    if not trim:
        res = res.reindex(p.returns.index)
    return res if p.multi else res.iloc[:, 0]
