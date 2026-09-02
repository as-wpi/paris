"""Drawdown series, episodes and drawdown-based risk measures.

Conventions:
* Drawdowns are measured from the running peak of the wealth index, which starts at 1, so the
  first observation can already be in drawdown. Values are <= 0 (``max_drawdown`` is negative).
* An episode runs from the first period below the peak to the period the peak is regained
  (inclusive). For an unrecovered drawdown PARIS counts periods to the last observation.
* The drawdown-based ratios annualise by observation count (``method="periods"``, as
  :func:`paris.annualized_return`); ``method="calendar"`` annualises over elapsed calendar days
  from ``start`` (default: the first return date), as :func:`paris.cagr`.
* ``window`` on the Calmar and Sterling ratios restricts numerator and denominator to the trailing
  ``window`` observations (Young 1991: 36 months); the default is the whole history.
* The Ulcer Index is a decimal (``pct=True`` reports percent, as in Martin's original).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from paris._core import annualize_return, prepare, result, rf_annual, safe_div

__all__ = [
    "avg_drawdown",
    "avg_drawdown_length",
    "avg_recovery",
    "burke_ratio",
    "calmar_ratio",
    "current_drawdown",
    "drawdown_deviation",
    "drawdown_distribution",
    "drawdown_table",
    "drawdowns",
    "longest_drawdown",
    "martin_ratio",
    "max_drawdown",
    "pain_index",
    "pain_ratio",
    "recovery_factor",
    "rolling_ulcer",
    "sterling_ratio",
    "ulcer_index",
    "ulcer_performance_index",
]


def _tail(s: pd.Series, window: int | None) -> pd.Series:
    """The trailing ``window`` observations (validated), or the whole series."""
    if window is None:
        return s
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must be an integer >= 2")
    if window > len(s):
        raise ValueError(f"window ({window}) exceeds the {len(s)} observations")
    return s.iloc[-window:]


def _ann(s: pd.Series, ppy: int, geometric: bool, method: str, days_in_year: float,
         start: Any) -> float:
    """Annualised return by observation count (``periods``) or elapsed calendar days (``calendar``).

    Ecosystem-first note: the drawdown ratios below are PARIS's own pure numpy/pandas
    implementations (the package has no empyrical dependency by design); the default
    ``method="periods"`` Calmar matches ``empyrical.calmar_ratio`` to machine precision.
    """
    if method == "periods":
        return annualize_return(s, ppy, geometric)
    if method != "calendar":
        raise ValueError("method must be 'periods' or 'calendar'")
    first = pd.Timestamp(start) if start is not None else s.index[0]
    days = (s.index[-1] - first).days
    if days <= 0:
        return float("nan")
    if geometric:
        return float(np.prod(1.0 + s.values) ** (days_in_year / days) - 1.0)
    return float(s.sum() * days_in_year / days)


def _dd(s: pd.Series, geometric: bool) -> pd.Series:
    wealth = (1 + s).cumprod() if geometric else 1 + s.cumsum()
    peak = np.maximum.accumulate(np.r_[1.0, wealth.values])[1:]
    return wealth / peak - 1


def _unwrap(df: pd.DataFrame, multi: bool) -> Any:
    return df if multi else df.iloc[:, 0]


def drawdowns(returns: Any, geometric: bool = True) -> Any:
    """Drawdown from running peak per period (<= 0)."""
    p = prepare(returns, periods_per_year=1)
    return _unwrap(p.returns.apply(_dd, geometric=geometric), p.multi)


def max_drawdown(returns: Any, geometric: bool = True) -> Any:
    """Deepest drawdown as a negative number."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(_dd(s, geometric).min()))


def episodes(dd: pd.Series) -> pd.DataFrame:
    """Drawdown episodes of a drawdown series: integer positions ``start, trough, end`` (end = first
    period back at the peak, or NaN if unrecovered), ``depth``, ``length``, ``to_trough``, ``recovery``.
    """
    v = dd.values
    rows = []
    i, n = 0, len(v)
    while i < n:
        if v[i] >= 0:
            i += 1
            continue
        start = i
        while i < n and v[i] < 0:
            i += 1
        end = i if i < n else None  # i is the first non-negative period (recovery)
        seg = v[start:i]
        trough = start + int(np.argmin(seg))
        last = end if end is not None else n - 1
        rows.append({"start": start, "trough": trough, "end": end, "depth": float(seg.min()),
                     "length": last - start + 1, "to_trough": trough - start + 1,
                     "recovery": (end - trough) if end is not None else np.nan})
    cols = ["start", "trough", "end", "depth", "length", "to_trough", "recovery"]
    return pd.DataFrame(rows, columns=cols)


def _table(s: pd.Series, top: int, geometric: bool) -> pd.DataFrame:
    dd = _dd(s, geometric)
    ep = episodes(dd).sort_values("depth", kind="stable").head(top)
    idx = dd.index
    out = pd.DataFrame({
        "start": [idx[i] for i in ep["start"]],
        "trough": [idx[i] for i in ep["trough"]],
        "end": [idx[int(i)] if pd.notna(i) else pd.NaT for i in ep["end"]],
        "depth": ep["depth"].values,
        "length": ep["length"].values,
        "to_trough": ep["to_trough"].values,
        "recovery": ep["recovery"].values,
    })
    return out.reset_index(drop=True)


def drawdown_table(returns: Any, top: int = 5, geometric: bool = True) -> pd.DataFrame:
    """Worst ``top`` drawdowns: start, trough, end, depth, length (periods),
    to_trough, recovery. A DataFrame input yields a long table with a leading ``fund`` column."""
    p = prepare(returns, periods_per_year=1)
    if not p.multi:
        return _table(p.returns.iloc[:, 0], top, geometric)
    parts = []
    for col in p.returns.columns:
        t = _table(p.returns[col], top, geometric)
        t.insert(0, "fund", col)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def _episode_stat(returns: Any, field: str, geometric: bool = True) -> Any:
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        ep = episodes(_dd(s, geometric))
        return float(np.abs(ep[field]).mean()) if len(ep) else float("nan")
    return result(p, fn)


def avg_drawdown(returns: Any, geometric: bool = True) -> Any:
    """Mean depth of all drawdown episodes, as a positive number."""
    return _episode_stat(returns, "depth", geometric)


def avg_drawdown_length(returns: Any, geometric: bool = True) -> Any:
    """Mean episode length in periods (see module note on unrecovered)."""
    return _episode_stat(returns, "length", geometric)


def avg_recovery(returns: Any, geometric: bool = True) -> Any:
    """Mean periods from trough to recovery over recovered episodes (unrecovered episodes are
    excluded)."""
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        rec = episodes(_dd(s, geometric))["recovery"].dropna()
        return float(rec.mean()) if len(rec) else float("nan")
    return result(p, fn)


def drawdown_deviation(returns: Any, geometric: bool = True) -> Any:
    """``sqrt(sum(depth_i^2) / n)`` over episodes, n = observations."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(np.sqrt((episodes(_dd(s, geometric))["depth"] ** 2).sum() / len(s))))


def ulcer_index(returns: Any, geometric: bool = True, ddof: int = 0, pct: bool = False) -> Any:
    """Root-mean-square of the drawdown series (Martin's Ulcer Index); ``ddof=1`` divides by
    ``n-1``; ``pct=True`` reports it in percent."""
    p = prepare(returns, periods_per_year=1)
    scale = 100.0 if pct else 1.0
    return result(p, lambda s: scale * float(np.sqrt((_dd(s, geometric) ** 2).sum() / (len(s) - ddof))))


def rolling_ulcer(returns: Any, window: int | None = None, geometric: bool = True, ddof: int = 0,
                  pct: bool = False, trim: bool = True, periods_per_year: int | None = None) -> Any:
    """Ulcer Index over a sliding window of ``window`` observations (default: one year, i.e. the
    periods per year): the root-mean-square of the drawdowns inside each window, where every
    drawdown is measured from the running peak of the whole history (not re-based inside the
    window). Equals ``rolling(returns, ulcer_index, window)`` only when no window starts inside a
    drawdown. Series in, Series out; DataFrame in, one column per fund; rows before the first
    complete window are dropped (``trim=False`` keeps them as NaN).
    """
    p = prepare(returns, periods_per_year=1 if window is not None else periods_per_year)
    n = len(p.returns)
    if window is None:
        window = p.ppy
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must be an integer >= 2")
    if window > n:
        raise ValueError(f"window ({window}) exceeds the {n} observations")
    dd2 = p.returns.apply(_dd, geometric=geometric) ** 2
    # the sliding-window sum leaves residues of order 1e-19 when a drawdown leaves the window;
    # clamp so that a window at the peak is exactly 0, never sqrt(-1e-19) = NaN
    ui = np.sqrt(np.maximum(dd2.rolling(window).sum(), 0.0) / (window - ddof)) * (100.0 if pct else 1.0)
    if trim:
        ui = ui.iloc[window - 1:]
    return _unwrap(ui, p.multi)


def drawdown_distribution(returns: Any, stat: str = "drawdown", window: int | None = None,
                          quantiles: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99),
                          threshold: float = 0.01, geometric: bool = True, ddof: int = 0,
                          periods_per_year: int | None = None) -> pd.DataFrame:
    """Distribution table of the per-period drawdowns (``stat="drawdown"``, the inputs to the Ulcer
    Index) or of the rolling Ulcer Index (``stat="ulcer"``, :func:`rolling_ulcer` over ``window``
    observations, one year by default): the share of observations smaller than ``threshold`` in
    magnitude, the mean, the ``quantiles`` of the magnitude and the maximum. Drawdown rows keep
    the negative sign (``q95`` is the 95th percentile of severity); Ulcer rows are positive.
    Rows = statistics, columns = funds."""
    if stat not in ("drawdown", "ulcer"):
        raise ValueError("stat must be 'drawdown' or 'ulcer'")
    p = prepare(returns, periods_per_year=1 if stat == "drawdown" else periods_per_year)
    if stat == "drawdown":
        x = p.returns.apply(_dd, geometric=geometric)
        sign = -1.0
    else:
        w = p.ppy if window is None else window
        x = rolling_ulcer(p.returns, w, geometric=geometric, ddof=ddof)
        sign = 1.0
    mag = x.abs()
    rows = {f"share |x| < {threshold:g}": (mag < threshold).mean(), "mean": sign * mag.mean()}
    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError("quantiles must lie in (0, 1)")
        rows[f"q{q * 100:g}"] = sign * mag.quantile(q)
    rows["max"] = sign * mag.max()
    out = pd.DataFrame(rows).T.astype(float)
    out.index.name = "statistic"
    return out


def pain_index(returns: Any, geometric: bool = True) -> Any:
    """Mean absolute drawdown (pain index)."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(_dd(s, geometric).abs().mean()))


def longest_drawdown(returns: Any, geometric: bool = True) -> Any:
    """Length in periods of the longest drawdown episode."""
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        ep = episodes(_dd(s, geometric))
        return float(ep["length"].max()) if len(ep) else 0.0
    return result(p, fn)


def current_drawdown(returns: Any, geometric: bool = True) -> Any:
    """Drawdown at the last observation."""
    p = prepare(returns, periods_per_year=1)
    return result(p, lambda s: float(_dd(s, geometric).iloc[-1]))


# --------------------------------------------------------------------------- ratios
def recovery_factor(returns: Any, geometric: bool = True) -> Any:
    """Total return / |max drawdown| (recovery factor)."""
    p = prepare(returns, periods_per_year=1)

    def fn(s: pd.Series) -> float:
        tot = float(np.prod(1 + s.values) - 1) if geometric else float(s.sum())
        return safe_div(tot, abs(float(_dd(s, geometric).min())))
    return result(p, fn)


def calmar_ratio(returns: Any, periods_per_year: int | None = None, geometric: bool = True,
                 window: int | None = None, method: str = "periods", days_in_year: float = 365.25,
                 start: Any = None) -> Any:
    """Annualised return / |max drawdown| (Calmar ratio). ``window`` restricts both to the trailing
    ``window`` observations (Young 1991: 36 months); ``method="calendar"`` annualises over elapsed
    calendar days from ``start`` (ignored with ``window``), as :func:`paris.cagr`."""
    p = prepare(returns, periods_per_year=periods_per_year)

    def fn(s: pd.Series) -> float:
        s = _tail(s, window)
        return safe_div(_ann(s, p.ppy, geometric, method, days_in_year, None if window else start),
                        abs(float(_dd(s, geometric).min())))
    return result(p, fn)


def sterling_ratio(returns: Any, excess: float = 0.10, periods_per_year: int | None = None,
                   geometric: bool = True, window: int | None = None, method: str = "periods",
                   days_in_year: float = 365.25, start: Any = None) -> Any:
    """Annualised return / (|max drawdown| + ``excess``) (Sterling ratio, ``excess`` 10 % by default).
    ``window``, ``method``, ``days_in_year`` and ``start`` as :func:`calmar_ratio`."""
    p = prepare(returns, periods_per_year=periods_per_year)

    def fn(s: pd.Series) -> float:
        s = _tail(s, window)
        return safe_div(_ann(s, p.ppy, geometric, method, days_in_year, None if window else start),
                        abs(float(_dd(s, geometric).min())) + excess)
    return result(p, fn)


def burke_ratio(returns: Any, rf: Any = 0.0, modified: bool = False,
                periods_per_year: int | None = None, geometric: bool = True, method: str = "periods",
                days_in_year: float = 365.25, start: Any = None) -> Any:
    """(Annualised return - rf) / sqrt(sum(depth_i^2)) over drawdown episodes; ``modified`` scales
    by sqrt(n). ``rf`` is an annual rate (a per-period Series is annualised geometrically).
    ``method="calendar"`` annualises the return over elapsed calendar days (see module note)."""
    p = prepare(returns, periods_per_year=periods_per_year)
    rfa = rf_annual(rf, p.returns.index, p.ppy)

    def fn(s: pd.Series) -> float:
        ep = episodes(_dd(s, geometric))
        denom = float(np.sqrt((ep["depth"] ** 2).sum()))
        r = (_ann(s, p.ppy, geometric, method, days_in_year, start) - rfa) / denom if denom else float("nan")
        return r * np.sqrt(len(s)) if modified else r
    return result(p, fn)


def pain_ratio(returns: Any, rf: Any = 0.0, periods_per_year: int | None = None,
               geometric: bool = True, method: str = "periods", days_in_year: float = 365.25,
               start: Any = None) -> Any:
    """(Annualised return - annual rf) / pain index (pain ratio). A per-period rf Series is
    annualised geometrically. ``method="calendar"`` annualises over elapsed calendar days."""
    p = prepare(returns, periods_per_year=periods_per_year)
    rfa = rf_annual(rf, p.returns.index, p.ppy)
    return result(p, lambda s: safe_div(_ann(s, p.ppy, geometric, method, days_in_year, start) - rfa,
                                        float(_dd(s, geometric).abs().mean())))


def martin_ratio(returns: Any, rf: Any = 0.0, periods_per_year: int | None = None,
                 geometric: bool = True, ddof: int = 0, method: str = "periods",
                 days_in_year: float = 365.25, start: Any = None) -> Any:
    """(Annualised return - annual rf) / ulcer index (Martin ratio). A per-period rf Series is
    annualised geometrically. ``method="calendar"`` annualises over elapsed calendar days from
    ``start`` (default: the first return date; pass the date of the initial value to match a
    price-based calculation)."""
    p = prepare(returns, periods_per_year=periods_per_year)
    rfa = rf_annual(rf, p.returns.index, p.ppy)
    return result(p, lambda s: safe_div(_ann(s, p.ppy, geometric, method, days_in_year, start) - rfa,
                                        float(np.sqrt((_dd(s, geometric) ** 2).sum() / (len(s) - ddof)))))


def ulcer_performance_index(returns: Any, rf: Any = 0.0, periods_per_year: int | None = None,
                            annualize: bool = True, ddof: int = 0) -> Any:
    """Ulcer performance index. Annualised (== :func:`martin_ratio`) by default;
    ``annualize=False`` gives ``(total_return - rf) / ulcer_index``.
    ``rf`` is an annual scalar or a per-period Series (annualised geometrically)."""
    if annualize:
        return martin_ratio(returns, rf, periods_per_year, ddof=ddof)
    p = prepare(returns, periods_per_year=1)
    rfa = rf_annual(rf, p.returns.index, periods_per_year)
    return result(p, lambda s: safe_div(float(np.prod(1 + s.values) - 1) - rfa,
                                        float(np.sqrt((_dd(s, True) ** 2).sum() / (len(s) - ddof)))))
