"""One-call summary table of all PARIS metrics, plus the metric registry used by tables/charts."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from paris import drawdown as D
from paris import ratios as Q
from paris import relative as L
from paris import returns as R
from paris import risk as K
from paris._core import prepare

__all__ = ["ABSOLUTE_METRICS", "RELATIVE_METRICS", "stats"]

# label -> callable(returns, rf=, periods_per_year=) for metrics defined on a single series.
ABSOLUTE_METRICS: dict[str, Callable[..., Any]] = {
    "Start": lambda r, **k: None,  # filled separately
    "End": lambda r, **k: None,
    "Periods": lambda r, **k: None,
    "Total Return": lambda r, **k: R.total_return(r),
    "CAGR": lambda r, rf, ppy: R.annualized_return(r, ppy),
    "Volatility (ann.)": lambda r, rf, ppy: K.volatility(r, ppy),
    "Semi Deviation": lambda r, **k: K.semi_deviation(r),
    "Gain Deviation": lambda r, **k: K.gain_deviation(r),
    "Loss Deviation": lambda r, **k: K.loss_deviation(r),
    "Sharpe": lambda r, rf, ppy: Q.sharpe(r, rf, ppy),
    "Sortino": lambda r, rf, ppy: Q.sortino(r, periods_per_year=ppy),
    "Calmar": lambda r, rf, ppy: D.calmar_ratio(r, ppy),
    "Max Drawdown": lambda r, **k: D.max_drawdown(r),
    "Longest Drawdown (periods)": lambda r, **k: D.longest_drawdown(r),
    "Average Drawdown": lambda r, **k: D.avg_drawdown(r),
    "Average Length (periods)": lambda r, **k: D.avg_drawdown_length(r),
    "Average Recovery (periods)": lambda r, **k: D.avg_recovery(r),
    "Current Drawdown": lambda r, **k: D.current_drawdown(r),
    "Ulcer Index": lambda r, **k: D.ulcer_index(r),
    "Martin": lambda r, rf, ppy: D.martin_ratio(r, periods_per_year=ppy),
    "Sterling": lambda r, rf, ppy: D.sterling_ratio(r, periods_per_year=ppy),
    "Burke": lambda r, rf, ppy: D.burke_ratio(r, periods_per_year=ppy),
    "Pain Ratio": lambda r, rf, ppy: D.pain_ratio(r, periods_per_year=ppy),
    "Pain Index": lambda r, **k: D.pain_index(r),
    "Omega": lambda r, **k: Q.omega(r),
    "Skewness": lambda r, **k: K.skewness(r),
    "Kurtosis (excess)": lambda r, **k: K.kurtosis(r),
    "VaR 95% (hist.)": lambda r, **k: K.var(r),
    "CVaR 95% (hist.)": lambda r, **k: K.cvar(r),
    "VaR 95% (modified)": lambda r, **k: K.var(r, method="modified"),
    "Best Period": lambda r, **k: R.best(r),
    "Worst Period": lambda r, **k: R.worst(r),
    "Win Rate": lambda r, **k: R.win_rate(r),
    "Payoff Ratio": lambda r, **k: Q.payoff_ratio(r),
    "Profit Factor": lambda r, **k: Q.profit_factor(r),
    "Tail Ratio": lambda r, **k: K.tail_ratio(r),
    "Kelly leverage (half)": lambda r, rf, ppy: Q.kelly_ratio(r, rf, ppy, half=True),
}

RELATIVE_METRICS: dict[str, Callable[..., Any]] = {
    "Beta": lambda r, b, rf, ppy: L.beta(r, b, rf, ppy),
    "Alpha (ann.)": lambda r, b, rf, ppy: L.alpha(r, b, rf, ppy),
    "Jensen Alpha": lambda r, b, rf, ppy: L.jensen_alpha(r, b, rf, ppy),
    "R-squared": lambda r, b, rf, ppy: L.r_squared(r, b, rf),
    "Correlation": lambda r, b, rf, ppy: L.correlation(r, b),
    "Tracking Error": lambda r, b, rf, ppy: L.tracking_error(r, b, ppy),
    "Information Ratio": lambda r, b, rf, ppy: L.information_ratio(r, b, ppy),
    "Active Premium": lambda r, b, rf, ppy: L.active_premium(r, b, ppy),
    "Treynor": lambda r, b, rf, ppy: L.treynor_ratio(r, b, rf, ppy),
    "M-squared": lambda r, b, rf, ppy: L.m_squared(r, b, rf, ppy),
    "Up Capture": lambda r, b, rf, ppy: L.up_capture(r, b),
    "Down Capture": lambda r, b, rf, ppy: L.down_capture(r, b),
    "Up Number Ratio": lambda r, b, rf, ppy: L.up_number_ratio(r, b),
    "Down Number Ratio": lambda r, b, rf, ppy: L.down_number_ratio(r, b),
    "Up Percentage Ratio": lambda r, b, rf, ppy: L.up_percentage_ratio(r, b),
    "Down Percentage Ratio": lambda r, b, rf, ppy: L.down_percentage_ratio(r, b),
    "Batting Average": lambda r, b, rf, ppy: L.batting_average(r, b),
    "Bull Beta": lambda r, b, rf, ppy: L.bull_beta(r, b, rf, ppy),
    "Bear Beta": lambda r, b, rf, ppy: L.bear_beta(r, b, rf, ppy),
}


def stats(returns: Any, benchmark: Any = None, rf: Any = 0.0, periods_per_year: int | None = None,
          include_benchmark: bool = True, metrics: list[str] | None = None) -> pd.DataFrame:
    """Summary table: one row per metric, one column per fund (and the benchmark, if given).

    All inputs are aligned to the common window first (see :func:`paris._core.align`). ``rf`` is an
    annual rate or a per-period Series; ``metrics`` restricts the rows (labels from
    ``ABSOLUTE_METRICS`` / ``RELATIVE_METRICS``).
    """
    p = prepare(returns, benchmark=benchmark, rf=rf, periods_per_year=periods_per_year)
    df, ppy = p.returns, p.ppy
    rfs = p.rf
    cols = {c: df[c] for c in df.columns}
    if p.benchmark is not None and include_benchmark:
        cols[p.benchmark.name or "benchmark"] = p.benchmark
    table: dict[str, dict[str, Any]] = {}
    for label, fn in ABSOLUTE_METRICS.items():
        if metrics is not None and label not in metrics:
            continue
        row = {}
        for name, s in cols.items():
            if label == "Start":
                row[name] = s.index[0]
            elif label == "End":
                row[name] = s.index[-1]
            elif label == "Periods":
                row[name] = len(s)
            else:
                row[name] = fn(s, rf=rfs, ppy=ppy)
        table[label] = row
    if p.benchmark is not None:
        for label, fn in RELATIVE_METRICS.items():
            if metrics is not None and label not in metrics:
                continue
            row = {name: fn(s, p.benchmark, rfs, ppy) for name, s in cols.items() if name in df.columns}
            if include_benchmark:
                bname = p.benchmark.name or "benchmark"
                row[bname] = fn(p.benchmark.rename(bname), p.benchmark, rfs, ppy)
            table[label] = row
    out = pd.DataFrame(table).T
    out.index.name = "metric"
    return out
