"""Portfolio construction from weights, contributions to return, and Brinson attribution.

Conventions:
* **Weights** are a one-time vector (list / array in column order, dict / Series by asset) or a
  DataFrame of dated rows. A weight row dated *d* is effective for the returns **strictly after**
  *d* (the trade happens at the close of *d*); a vector is dated one day before the first return
  and so covers every period. Returns before the first weight date are dropped. Every row must sum to 1 within ``1e-4`` unless
  ``normalize=True`` rescales it; the assets it names are the portfolio (other columns are
  dropped).
* **Drift.** Between weight dates, holdings are buy-and-hold: the value of each asset compounds
  with its own return and the weights drift. ``rebalance`` (a pandas offset alias such as ``"ME"``,
  ``"QE"``, ``"YE"``, ``"W"``) resets a one-time vector to target after the last observation of
  each period; rebalancing at the data's own frequency gives constant weights.
* **Contribution** of an asset in a period is its beginning-of-period weight times its return, so
  the contributions sum to the portfolio return. Multi-period contributions ([[period_contributions]])
  scale each period's contribution by the portfolio wealth at the start of the period relative
  to the start of the span, so they sum exactly to the span's compounded return.
* **Brinson** effects per asset and period use the Brinson–Fachler form by default
  (``method="BF"``, allocation measured against the benchmark's total return); ``"BHB"`` is the
  Brinson–Hood–Beebower form (allocation against zero). Periods are linked with the Carino
  log-linking coefficients by default (effects sum exactly to the geometric active return);
  ``"menchero"`` uses the Menchero coefficients and ``"none"`` is the arithmetic sum. An asset absent from one side takes the other side's
  return for that period (weight 0), so its whole effect is allocation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from paris._core import AlignmentError, align, to_frame, to_weights

__all__ = [
    "active_contribution",
    "bop_weights",
    "brinson",
    "contribution",
    "eop_weights",
    "period_contributions",
    "portfolio_return",
]

EFFECTS = ["Allocation", "Selection", "Interaction", "Active"]


# --------------------------------------------------------------------------- simulation
def _simulate(returns: Any, weights: Any, rebalance: str | None, normalize: bool
              ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Value-drift loop of the weighted portfolio (geometric).

    Returns (asset returns used, weight table, portfolio returns, contributions, BOP weights,
    EOP weights); the arrays are rows = the returns kept (after the first weight date).
    """
    df, _, _ = align(returns)
    wtab = to_weights(weights, df.columns, df.index, normalize=normalize)
    df = df[wtab.columns]
    if rebalance is not None:
        if isinstance(weights, pd.DataFrame) or len(wtab) != 1:
            raise ValueError("rebalance applies to a one-time weight vector, not a dated weight table")
        ends = df.index.to_series().resample(rebalance).last().dropna()
        dates = pd.DatetimeIndex([wtab.index[0], *ends.values]).unique()
        wtab = pd.DataFrame(np.tile(wtab.values, (len(dates), 1)), index=dates, columns=wtab.columns)
    df = df[df.index > wtab.index[0]]
    if df.empty:
        raise AlignmentError("no returns after the first weight date")
    row = np.searchsorted(wtab.index.values, df.index.values, side="left") - 1   # last weight date < t
    r, w = df.values, wtab.values
    n, k = r.shape
    ret, contrib, bop, eop = np.empty(n), np.empty((n, k)), np.empty((n, k)), np.empty((n, k))
    end_value, prev_eop_value = 1.0, None
    with np.errstate(all="ignore"):           # a total loss (r = -1) leaves 0/0 weights: NaN, silently
        for t in range(n):
            prior = end_value
            if t == 0 or row[t] != row[t - 1]:
                value = w[row[t]] * end_value                   # (re)set to the effective weights
            else:
                value = prev_eop_value                          # drift
            bop[t] = value / value.sum()
            contrib[t] = r[t] * value / prior
            prev_eop_value = value * (1.0 + r[t])
            end_value = prev_eop_value.sum()
            eop[t] = prev_eop_value / end_value
            ret[t] = end_value / prior - 1.0
    return df, wtab, ret, contrib, bop, eop


def portfolio_return(returns: Any, weights: Any, *, rebalance: str | None = None,
                     normalize: bool = False) -> pd.Series:
    """Periodic returns of the weighted portfolio."""
    df, _, ret, *_ = _simulate(returns, weights, rebalance, normalize)
    return pd.Series(ret, index=df.index, name="Portfolio")


def contribution(returns: Any, weights: Any, *, rebalance: str | None = None,
                 normalize: bool = False) -> pd.DataFrame:
    """Per-period contribution of each asset: BOP weight × return; rows sum to the portfolio return."""
    df, _, _, contrib, *_ = _simulate(returns, weights, rebalance, normalize)
    return pd.DataFrame(contrib, index=df.index, columns=df.columns)


def bop_weights(returns: Any, weights: Any, *, rebalance: str | None = None,
                normalize: bool = False) -> pd.DataFrame:
    """Beginning-of-period weights actually held (target after a weight date, drifted otherwise)."""
    df, _, _, _, bop, _ = _simulate(returns, weights, rebalance, normalize)
    return pd.DataFrame(bop, index=df.index, columns=df.columns)


def eop_weights(returns: Any, weights: Any, *, rebalance: str | None = None,
                normalize: bool = False) -> pd.DataFrame:
    """End-of-period weights after each period's returns (before any rebalancing trade)."""
    df, _, _, _, _, eop = _simulate(returns, weights, rebalance, normalize)
    return pd.DataFrame(eop, index=df.index, columns=df.columns)


# --------------------------------------------------------------------------- linking
def period_contributions(contributions: Any, freq: str | None = None) -> Any:
    """Link per-period contributions to a span.

    Each period's contribution is scaled by the portfolio wealth at the start of that period
    relative to the start of its span, so the linked contributions of a span sum to its compounded
    portfolio return (the ``Portfolio`` column). ``freq=None`` links the whole window and returns a
    Series by asset; a pandas offset alias (``"YE"``, ``"QE"``, ...) gives one row per span, indexed
    by the span's last observation.
    """
    c = to_frame(contributions, name="contribution")
    wealth = (1.0 + c.sum(axis=1)).cumprod()
    lag = wealth.shift(1).fillna(1.0)
    scaled = c.mul(lag, axis=0)
    if freq is None:
        out = scaled.sum()
        out["Portfolio"] = out.sum()
        return out.rename("contribution")
    rows, idx = [], []
    for _, sub in scaled.groupby(pd.Grouper(freq=freq)):
        if len(sub):
            rows.append(sub.sum() / lag.loc[sub.index[0]])
            idx.append(sub.index[-1])
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name=c.index.name))
    out["Portfolio"] = out.sum(axis=1)
    return out


# --------------------------------------------------------------------------- relative
def _pair(returns, weights, benchmark_returns, benchmark_weights, rebalance, benchmark_rebalance,
          normalize):
    """Both simulations on their common dates; asset union in portfolio-then-benchmark order."""
    p = _simulate(returns, weights, rebalance, normalize)
    b = _simulate(benchmark_returns, benchmark_weights, benchmark_rebalance, normalize)
    start = max(p[0].index[0], b[0].index[0])
    end = min(p[0].index[-1], b[0].index[-1])
    if start > end:
        raise AlignmentError("portfolio and benchmark share no common window")
    ip = (p[0].index >= start) & (p[0].index <= end)
    ib = (b[0].index >= start) & (b[0].index <= end)
    if not p[0].index[ip].equals(b[0].index[ib]):
        raise AlignmentError("portfolio and benchmark dates differ inside the common window")
    assets = list(p[0].columns) + [a for a in b[0].columns if a not in p[0].columns]
    return p, b, ip, ib, assets


def active_contribution(returns: Any, weights: Any, benchmark_returns: Any, benchmark_weights: Any, *,
                        rebalance: str | None = None, benchmark_rebalance: str | None = None,
                        normalize: bool = False) -> pd.DataFrame:
    """Per-period contribution minus the benchmark portfolio's, per asset, on the union of assets.

    An asset held on one side only contributes its full (signed) contribution; rows sum to the
    active return (portfolio return − benchmark portfolio return) of the period.
    """
    p, b, ip, ib, assets = _pair(returns, weights, benchmark_returns, benchmark_weights,
                                 rebalance, benchmark_rebalance, normalize)
    cp = pd.DataFrame(p[3][ip], index=p[0].index[ip], columns=p[0].columns)
    cb = pd.DataFrame(b[3][ib], index=b[0].index[ib], columns=b[0].columns)
    return cp.reindex(columns=assets, fill_value=0.0) - cb.reindex(columns=assets, fill_value=0.0)


def _close(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise "equal up to rounding": the linking ratios below take their analytic limit there."""
    return np.abs(x - y) <= 1e-10 * np.maximum(1.0, np.maximum(np.abs(x), np.abs(y)))


def _log_ratio(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Carino's (ln(1+x) − ln(1+y)) / (x − y), with the limit 1/(1+x) when x ≈ y."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(_close(x, y), 1.0 / (1.0 + 0.5 * (x + y)), (np.log1p(x) - np.log1p(y)) / (x - y))


def _link(effects: np.ndarray, rp: np.ndarray, rb: np.ndarray, linking: str) -> np.ndarray:
    """Combine per-period effect arrays (T × assets × 3) into one (assets × 3)."""
    if linking == "none":
        return effects.sum(axis=0)
    big_p, big_b, T = np.prod(1.0 + rp) - 1.0, np.prod(1.0 + rb) - 1.0, len(rp)
    if min(big_p, big_b, rp.min(), rb.min()) <= -1.0:
        return np.full(effects.shape[1:], np.nan)                       # a total loss has no log-link
    if linking == "carino":
        coef = _log_ratio(rp, rb) / _log_ratio(np.array(big_p), np.array(big_b))
    else:                                                               # menchero (validated by brinson)
        a = rp - rb
        if _close(np.array(big_p), np.array(big_b)):
            scale = (1.0 + 0.5 * (big_p + big_b)) ** ((T - 1.0) / T)
        else:
            scale = (big_p - big_b) / T / ((1.0 + big_p) ** (1.0 / T) - (1.0 + big_b) ** (1.0 / T))
        ss = (a ** 2).sum()
        c = 0.0 if ss == 0 else (big_p - big_b - scale * a.sum()) / ss
        coef = scale + c * a
    return np.tensordot(coef, effects, axes=(0, 0))


def brinson(returns: Any, weights: Any, benchmark_returns: Any, benchmark_weights: Any, *,
            rebalance: str | None = None, benchmark_rebalance: str | None = None,
            normalize: bool = False, method: str = "BF", linking: str = "carino",
            freq: str | None = None) -> pd.DataFrame:
    """Brinson attribution of the active return into allocation, selection and interaction.

    Per period and asset, with portfolio / benchmark BOP weights ``wp, wb`` and returns ``rp, rb``
    and the benchmark total ``Rb``: allocation ``(wp − wb)(rb − Rb)`` (``method="BF"``; ``"BHB"``
    uses ``(wp − wb) rb``), selection ``wb (rp − rb)``, interaction ``(wp − wb)(rp − rb)``.
    Periods are linked with ``linking`` = ``"carino"`` (default; sums to the geometric active
    return), ``"menchero"`` or ``"none"`` (arithmetic sum). Rows = assets plus ``Total``, columns
    ``Allocation, Selection, Interaction, Active``; with ``freq`` a MultiIndex (period, asset).
    """
    if method not in ("BF", "BHB"):
        raise ValueError("method must be 'BF' (Brinson-Fachler) or 'BHB' (Brinson-Hood-Beebower)")
    if linking not in ("carino", "menchero", "none"):
        raise ValueError("linking must be 'carino', 'menchero' or 'none'")
    p, b, ip, ib, assets = _pair(returns, weights, benchmark_returns, benchmark_weights,
                                 rebalance, benchmark_rebalance, normalize)
    idx = p[0].index[ip]
    rp_a = p[0][ip].reindex(columns=assets)
    rb_a = b[0][ib].reindex(columns=assets)
    rp_a, rb_a = rp_a.fillna(rb_a).values, rb_a.fillna(rp_a).values     # absent side: other side's return
    wp = pd.DataFrame(p[4][ip], columns=p[0].columns).reindex(columns=assets, fill_value=0.0).values
    wb = pd.DataFrame(b[4][ib], columns=b[0].columns).reindex(columns=assets, fill_value=0.0).values
    rp, rb = (wp * rp_a).sum(axis=1), (wb * rb_a).sum(axis=1)
    alloc = (wp - wb) * (rb_a - (rb[:, None] if method == "BF" else 0.0))
    sel = wb * (rp_a - rb_a)
    inter = (wp - wb) * (rp_a - rb_a)
    effects = np.stack([alloc, sel, inter], axis=-1)                     # T × assets × 3

    def table(e: np.ndarray, r_p: np.ndarray, r_b: np.ndarray) -> pd.DataFrame:
        linked = _link(e, r_p, r_b, linking)
        out = pd.DataFrame(linked, index=assets, columns=EFFECTS[:3])
        out.loc["Total"] = out.sum(skipna=False)
        out["Active"] = out.sum(axis=1, skipna=False)
        out.index.name = "asset"
        return out

    if freq is None:
        return table(effects, rp, rb)
    parts = {}
    for _, sub in pd.Series(np.arange(len(idx)), index=idx).groupby(pd.Grouper(freq=freq)):
        if len(sub):
            pos = sub.values
            parts[idx[pos[-1]]] = table(effects[pos], rp[pos], rb[pos])
    return pd.concat(parts, names=["period", "asset"])
