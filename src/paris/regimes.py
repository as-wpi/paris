"""Market regimes from slow and fast time-series momentum (Goulding, Harvey & Mazzoleni, 2023).

Goulding, C. L., Harvey, C. R. & Mazzoleni, M. G. (2023). Momentum turning points. *Journal of
Financial Economics*, 149(3), 378-406. Two binary time-series-momentum signals classify every
period into one of four observable states:

    SLOW  = sign of the trailing ``slow``-period arithmetic mean return   (paper: 12 months)
    FAST  = sign of the trailing ``fast``-period arithmetic mean return   (paper: 1 month)

    Bull        SLOW >= 0 and FAST >= 0      sustained uptrend
    Correction  SLOW >= 0 and FAST <  0      possible turn from up to down (often a false alarm)
    Bear        SLOW <  0 and FAST <  0      sustained downtrend
    Rebound     SLOW <  0 and FAST >= 0      possible turn from down to up

Conventions:
* Ties (a trailing mean of zero) are nonnegative, as in the paper's equations (1)-(2). A trailing
  mean within 1e-15 of zero counts as zero: sliding-window sums leave residues of order 1e-19, and
  a zero-return period computed from rounded prices carries a residue of order 1e-16.
* ``basis`` selects the return series the signals are computed on: ``"raw"`` (the series itself),
  ``"excess"`` (minus ``rf``, the paper's construction) or ``"relative"`` (minus ``benchmark``;
  the bundled S&P 500 proxy ``SPY`` from :mod:`paris.data` when no benchmark is given).
* Signals are arithmetic means of period returns (the paper's definition); ``compound=True``
  uses the compounded trailing return instead. The two differ in sign only near zero.
* Default lookbacks follow the inferred frequency: 12/1 monthly, 252/21 daily, 52/4 weekly,
  4/1 quarterly. Pass ``slow`` / ``fast`` explicitly for any other frequency.
* A state at date *t* is observable at *t* and describes the trend *entering* *t+1*; the
  conditional tables therefore pair the state at *t* with the return at *t+1*.
* States are diagnostic. The paper's dynamic-speed strategy improves the Sharpe ratio of static
  momentum, but the conditional alphas are modest; nothing here is a trading rule.

This is the only topic module that imports :mod:`paris.data` (lazily, for the default benchmark).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from paris._core import AlignmentError, Prepared, prepare, resolve_periods, to_series

__all__ = [
    "dynamic_speeds",
    "momentum_signal",
    "momentum_speed_weights",
    "momentum_state",
    "momentum_state_age",
    "momentum_state_table",
    "momentum_states",
    "momentum_transitions",
]

STATES: tuple[str, ...] = ("Bull", "Correction", "Bear", "Rebound")
_CODES = {s: i for i, s in enumerate(STATES)}
_DEFAULT_LOOKBACKS: dict[int, tuple[int, int]] = {252: (252, 21), 52: (52, 4), 12: (12, 1), 4: (4, 1)}
_BASES = ("raw", "excess", "relative")


# --------------------------------------------------------------------------- preparation
def _default_benchmark(ppy: int) -> pd.Series:
    """The bundled S&P 500 total-return proxy at the matching frequency (monthly or daily)."""
    from paris import data  # lazy: keeps the topic modules free of a data dependency

    if ppy == 12:
        return data.load_managers()["SPY"]
    if ppy == 252:
        return data.load_prices()["SPY"].pct_change().dropna()
    raise ValueError("no bundled S&P 500 series at this frequency; pass benchmark explicitly")


def _lookbacks(ppy: int, slow: int | None, fast: int | None) -> tuple[int, int]:
    if slow is None or fast is None:
        if ppy not in _DEFAULT_LOOKBACKS:
            raise ValueError(f"no default lookbacks for {ppy} periods per year; pass slow and fast")
        d_slow, d_fast = _DEFAULT_LOOKBACKS[ppy]
        slow, fast = (d_slow if slow is None else slow), (d_fast if fast is None else fast)
    if not (isinstance(slow, (int, np.integer)) and isinstance(fast, (int, np.integer))):
        raise ValueError("slow and fast must be integers")
    if fast < 1 or slow <= fast:
        raise ValueError(f"need slow > fast >= 1, got slow={slow}, fast={fast}")
    return int(slow), int(fast)


def _signal_frame(returns: Any, basis: str, rf: Any, benchmark: Any,
                  periods_per_year: int | None) -> tuple[Prepared, pd.DataFrame]:
    """Align inputs and return (Prepared, frame of signal returns) for the chosen basis."""
    if basis not in _BASES:
        raise ValueError(f"basis must be one of {_BASES}")
    if basis == "excess":
        if rf is None:
            raise ValueError("basis='excess' needs rf (annual scalar or per-period Series)")
        p = prepare(returns, rf=rf, periods_per_year=periods_per_year)
        return p, p.excess
    if basis == "relative":
        if benchmark is None:
            ppy = resolve_periods(prepare(returns, periods_per_year=periods_per_year).returns.index,
                                  periods_per_year)
            benchmark = _default_benchmark(ppy)
        p = prepare(returns, benchmark=benchmark, periods_per_year=periods_per_year)
        return p, p.active
    p = prepare(returns, periods_per_year=periods_per_year)
    return p, p.returns


_ZERO = 1e-15  # sliding-window sums leave residues of order 1e-19; a tie must stay a tie


def _trailing(sig: pd.DataFrame, window: int, compound: bool) -> pd.DataFrame:
    if compound:
        x = np.expm1(np.log1p(sig).rolling(window).sum())
    else:
        x = sig.rolling(window).mean()
    return x.mask(x.abs() <= _ZERO, 0.0)  # NaN (warm-up) compares False and is kept


def _unwrap(df: pd.DataFrame, multi: bool) -> Any:
    return df if multi else df.iloc[:, 0]


def _classify(slow_sig: pd.DataFrame, fast_sig: pd.DataFrame) -> pd.DataFrame:
    """Object frame of state labels; NaN where either signal is still warming up."""
    sp, fp = slow_sig >= 0, fast_sig >= 0
    out = pd.DataFrame(index=slow_sig.index, columns=slow_sig.columns, dtype=object)
    for c in slow_sig.columns:
        labels = np.select([sp[c] & fp[c], sp[c] & ~fp[c], ~sp[c] & ~fp[c], ~sp[c] & fp[c]],
                           STATES, default=None).astype(object)
        labels[(slow_sig[c].isna() | fast_sig[c].isna()).to_numpy()] = np.nan
        out[c] = labels
    return out


def _states_frame(returns: Any, basis: str, rf: Any, benchmark: Any, periods_per_year: int | None,
                  slow: int | None, fast: int | None, compound: bool) -> tuple[Prepared, pd.DataFrame, pd.DataFrame]:
    p, sig = _signal_frame(returns, basis, rf, benchmark, periods_per_year)
    k_slow, k_fast = _lookbacks(p.ppy, slow, fast)
    if k_slow > len(sig):
        raise AlignmentError(f"slow lookback ({k_slow}) exceeds the {len(sig)} observations")
    states = _classify(_trailing(sig, k_slow, compound), _trailing(sig, k_fast, compound))
    return p, sig, states


# --------------------------------------------------------------------------- signals & states
def momentum_signal(returns: Any, signal: str = "slow", basis: str = "raw", rf: Any = None,
                    benchmark: Any = None, periods_per_year: int | None = None,
                    slow: int | None = None, fast: int | None = None, compound: bool = False) -> Any:
    """Trailing momentum signal per period: the ``"slow"`` or ``"fast"`` trailing mean (or
    compounded) return on the chosen ``basis``; NaN during the warm-up. Series in, Series out;
    DataFrame in, one column per fund."""
    if signal not in ("slow", "fast"):
        raise ValueError("signal must be 'slow' or 'fast'")
    p, sig = _signal_frame(returns, basis, rf, benchmark, periods_per_year)
    k_slow, k_fast = _lookbacks(p.ppy, slow, fast)
    return _unwrap(_trailing(sig, k_slow if signal == "slow" else k_fast, compound), p.multi)


def momentum_states(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                    periods_per_year: int | None = None, slow: int | None = None,
                    fast: int | None = None, compound: bool = False, codes: bool = False) -> Any:
    """State label per period (``"Bull"``, ``"Correction"``, ``"Bear"``, ``"Rebound"``), NaN during
    the ``slow`` warm-up. ``codes=True`` returns integers 0..3 in that order, -1 for warm-up."""
    p, _, states = _states_frame(returns, basis, rf, benchmark, periods_per_year, slow, fast, compound)
    if codes:
        states = states.apply(lambda s: s.map(_CODES).fillna(-1).astype(int))
    return _unwrap(states, p.multi)


def momentum_state(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                   periods_per_year: int | None = None, slow: int | None = None,
                   fast: int | None = None, compound: bool = False) -> Any:
    """The state at the last observation (a string; NaN if the history is too short). Series in,
    one label out; DataFrame in, a Series of labels by fund."""
    p, _, states = _states_frame(returns, basis, rf, benchmark, periods_per_year, slow, fast, compound)
    out = states.iloc[-1]
    return out if p.multi else out.iloc[0]


def momentum_state_age(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                       periods_per_year: int | None = None, slow: int | None = None,
                       fast: int | None = None, compound: bool = False) -> Any:
    """Number of consecutive periods (including the last) spent in the current state; NaN if the
    last state is undefined."""
    p, _, states = _states_frame(returns, basis, rf, benchmark, periods_per_year, slow, fast, compound)

    def age(s: pd.Series) -> float:
        vals = s.to_numpy(dtype=object)
        if vals[-1] is np.nan or vals[-1] is None or pd.isna(vals[-1]):
            return float("nan")
        n = 1
        for i in range(len(vals) - 2, -1, -1):
            if vals[i] != vals[-1]:
                break
            n += 1
        return float(n)

    out = pd.Series({c: age(states[c]) for c in states.columns}, dtype=float)
    return out if p.multi else out.iloc[0]


# --------------------------------------------------------------------------- conditional tables
def _next_by_state(states: pd.Series, sig: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Pairs (state at t, return at t+1) with both defined."""
    s, r = states.iloc[:-1], sig.iloc[1:]
    keep = s.notna().to_numpy()
    return s[keep].reset_index(drop=True), r[keep].reset_index(drop=True)


def _state_table(states: pd.Series, sig: pd.Series, ppy: int) -> pd.DataFrame:
    s, r = _next_by_state(states, sig)
    rows = {}
    for st in STATES:
        x = r[(s == st).to_numpy()]
        n = len(x)
        rows[st] = {
            "count": float(n),
            "frequency": n / len(s) if len(s) else float("nan"),
            "mean (ann.)": float(x.mean() * ppy) if n else float("nan"),
            "volatility (ann.)": float(x.std(ddof=1) * np.sqrt(ppy)) if n > 1 else float("nan"),
            "skewness": float(_pop_skew(x.to_numpy())) if n > 2 else float("nan"),
            "up frequency": float((x > 0).mean()) if n else float("nan"),
        }
    out = pd.DataFrame(rows).T.reindex(list(STATES))
    out.index.name = "state"
    return out


def _pop_skew(x: np.ndarray) -> float:
    m2 = np.mean((x - x.mean()) ** 2)
    return float("nan") if m2 == 0 else float(np.mean((x - x.mean()) ** 3) / m2 ** 1.5)


def _transitions(states: pd.Series) -> pd.DataFrame:
    s = states.dropna()
    a, b = s.iloc[:-1].to_numpy(), s.iloc[1:].to_numpy()
    out = pd.DataFrame(0.0, index=list(STATES), columns=list(STATES))
    for st in STATES:
        m = a == st
        n = int(m.sum())
        for nxt in STATES:
            out.loc[st, nxt] = float((b[m] == nxt).sum() / n) if n else float("nan")
    out.index.name, out.columns.name = "from", "to"
    return out


def _long(tables: dict[str, pd.DataFrame], multi: bool) -> pd.DataFrame:
    if not multi:
        return next(iter(tables.values()))
    parts = []
    for name, t in tables.items():
        t = t.reset_index()
        t.insert(0, "fund", name)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def momentum_state_table(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                         periods_per_year: int | None = None, slow: int | None = None,
                         fast: int | None = None, compound: bool = False) -> pd.DataFrame:
    """The paper's Figure 1 for one series: by state at *t*, the count and relative frequency of
    the state and the annualised mean, annualised volatility, population skewness and up-frequency
    of the *subsequent* return (on the same ``basis``). Rows = states; a DataFrame input gives one
    long table with a leading ``fund`` column."""
    p, sig, states = _states_frame(returns, basis, rf, benchmark, periods_per_year, slow, fast, compound)
    return _long({c: _state_table(states[c], sig[c], p.ppy) for c in states.columns}, p.multi)


def momentum_transitions(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                         periods_per_year: int | None = None, slow: int | None = None,
                         fast: int | None = None, compound: bool = False) -> pd.DataFrame:
    """The paper's Table 7: transition probabilities from the state at *t* (rows) to the state at
    *t+1* (columns); NaN row for a state never visited. A DataFrame input gives one long table with
    a leading ``fund`` column."""
    p, _, states = _states_frame(returns, basis, rf, benchmark, periods_per_year, slow, fast, compound)
    return _long({c: _transitions(states[c]) for c in states.columns}, p.multi)


# --------------------------------------------------------------------------- speed strategies
def _positions(states: pd.Series, a: float, speeds: dict[str, float] | None) -> pd.Series:
    """w_t(a) = (1 - a) w_SLOW + a w_FAST; Bull -> +1, Bear -> -1, Correction -> 1 - 2a,
    Rebound -> 2a - 1, with a state-specific ``a`` where ``speeds`` provides one."""
    sp = dict(speeds or {})
    a_co, a_re = sp.get("Correction", a), sp.get("Rebound", a)
    w = {"Bull": 1.0, "Bear": -1.0, "Correction": 1.0 - 2.0 * a_co, "Rebound": 2.0 * a_re - 1.0}
    return states.map(w).astype(float)


def momentum_speed_weights(returns: Any, a: float = 0.5, speeds: dict[str, float] | None = None,
                           basis: str = "raw", rf: Any = None, benchmark: Any = None,
                           periods_per_year: int | None = None, slow: int | None = None,
                           fast: int | None = None, compound: bool = False) -> Any:
    """Position of the intermediate-speed momentum strategy, equation (11):
    ``w_t(a) = (1 - a) w_SLOW,t + a w_FAST,t`` with ``w = +1`` for a nonnegative signal and ``-1``
    otherwise. ``a = 0`` is SLOW, ``a = 1`` FAST, ``a = 0.5`` the paper's MED. ``speeds`` maps
    ``"Correction"`` / ``"Rebound"`` to state-specific speeds (equation (30); Bull and Bear
    positions are +1 / -1 for every speed). The weight at *t* applies to the return at *t+1*;
    NaN during the warm-up."""
    if not 0.0 <= a <= 1.0:
        raise ValueError("a must lie in [0, 1]")
    for k, v in (speeds or {}).items():
        if k not in ("Correction", "Rebound"):
            raise ValueError("speeds keys must be 'Correction' and/or 'Rebound'")
        if not 0.0 <= v <= 1.0:
            raise ValueError("speeds must lie in [0, 1]")
    p, _, states = _states_frame(returns, basis, rf, benchmark, periods_per_year, slow, fast, compound)
    out = pd.DataFrame({c: _positions(states[c], a, speeds) for c in states.columns})
    return _unwrap(out, p.multi)


def _dynamic(states: pd.Series, sig: pd.Series) -> pd.Series:
    s, r = _next_by_state(states, sig)
    n = len(s)
    if n == 0:
        return pd.Series([np.nan, np.nan], index=["Correction", "Rebound"], dtype=float)
    sv, rv = s.to_numpy(), r.to_numpy()

    def m(st: str, power: int = 1) -> float:  # sample E[r^power | st] P[st]
        return float(np.sum(rv[sv == st] ** power) / n)

    def cond(st: str, power: int) -> float:  # sample E[r^power | st]
        x = rv[sv == st]
        return float(np.mean(x ** power)) if len(x) else float("nan")

    denom = m("Bull") - m("Bear")
    if not denom > 0:  # the first-order condition is then a minimiser: no Sharpe-maximising speed
        return pd.Series([np.nan, np.nan], index=["Correction", "Rebound"], dtype=float)
    q_bb = m("Bull", 2) + m("Bear", 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_co = 0.5 * (1.0 - q_bb / denom * cond("Correction", 1) / cond("Correction", 2))
        a_re = 0.5 * (1.0 + q_bb / denom * cond("Rebound", 1) / cond("Rebound", 2))
    return pd.Series([np.clip(a_co, 0.0, 1.0), np.clip(a_re, 0.0, 1.0)],
                     index=["Correction", "Rebound"], dtype=float)


def dynamic_speeds(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                   periods_per_year: int | None = None, slow: int | None = None,
                   fast: int | None = None, compound: bool = False) -> Any:
    """Sharpe-ratio-maximising state-dependent speeds, Proposition 9 (equations (32)-(33)):

    ``a_Co = 1/2 (1 - K E[r|Co] / E[r^2|Co])``, ``a_Re = 1/2 (1 + K E[r|Re] / E[r^2|Re])`` with
    ``K = E[r^2|Bull or Bear] P[Bull or Bear] / (E[r|Bull] P[Bull] - E[r|Bear] P[Bear])``, every
    moment a sample average of the return at *t+1* given the state at *t*, over the whole input.
    Estimates are clipped to [0, 1] (the paper's footnote 32); NaN when the maximiser condition
    ``E[r|Bull] P[Bull] > E[r|Bear] P[Bear]`` fails. Estimate on a training window and feed the
    result to :func:`momentum_speed_weights` (``speeds=``) to avoid look-ahead. Series in, a Series
    indexed ``Correction`` / ``Rebound``; DataFrame in, one column per fund."""
    p, sig, states = _states_frame(returns, basis, rf, benchmark, periods_per_year, slow, fast, compound)
    out = pd.DataFrame({c: _dynamic(states[c], sig[c]) for c in states.columns})
    return out if p.multi else out.iloc[:, 0]
