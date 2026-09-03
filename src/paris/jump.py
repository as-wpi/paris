"""Statistical jump models: causal two-state regime indicators (risk-on/off, trend-on/off).

The discrete statistical jump model (Nystrup, Lindström & Madsen 2020; Bemporad et al. 2018) fits
``K`` cluster centres to standardised features and a state sequence that minimises

    sum_t 1/2 ||x_t - mu_{s_t}||^2  +  lambda * (number of state changes),

by coordinate descent (dynamic programme for the sequence, means for the centres). The loss and
penalty conventions follow the reference implementation ``jumpmodels`` (Shu; used by Shu, Yu &
Mulvey 2024), so ``jump_penalty`` values are on the same scale. A pure numpy implementation is
kept here because PARIS carries no dependency beyond numpy and pandas; the ``oracle`` dependency
group installs ``jumpmodels`` and the test suite checks fit and online inference against it.

Causality conventions (every indicator in this module):
* Calibration is **rolling**: at each refit date the model (feature scaler + centres) is fitted on
  the ``window`` observations *before* that date, and the centres stay fixed until the next refit
  (``refit="ME"``: monthly; ``None``: a single fit on the first window).
* Inference is **online**: the state at *t* is the forward dynamic-programme argmin over the
  ``lookback`` observations ending at *t* (Shu, Yu & Mulvey 2024, Section 3.4.2) — no backward pass,
  nothing after *t* is used. ``jump_states`` reports that state at *t*.
* The indicators apply ``lag=1``: the value reported for day *T* is the online state at *T-1*, so
  it is known at the close of *T-1* and can act on *T*. The conditional tables pair the label at *T*
  with the return of *T* — a forecastable conditional mean.
* States are ordered by the centre of the **first** feature (ascending); the indicators map that to
  1 = risk-on (low volatility) and 1 = trend-on (high slow signal).
* Features are standardised with the training window's mean and standard deviation and clipped at
  ``clip`` standard deviations (``DataClipperStd`` in the reference), on the training window and on
  every lookback window alike.
* Warm-up (feature warm-up + ``window`` + ``lag``) is NaN, never filled.

Diagnostic, not a trading rule: online-inferred states are markedly noisier than the in-sample
path the same model draws with hindsight (Shu, Yu & Mulvey 2024, footnote 14).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from paris._core import AlignmentError, align, prepare, resolve_periods, to_frame
from paris.regimes import _lookbacks, _signal_frame, _trailing

__all__ = [
    "combine_states",
    "joint_state_table",
    "joint_states",
    "jump_labels",
    "jump_states",
    "risk_state_table",
    "risk_states",
    "state_table",
    "trend_state_table",
    "trend_states",
]

# Default jump penalties, sized on the daily US market factor (Fama-French, 1992-2026; 1,260-day
# window, monthly refit, lag 1; see CHANGELOG 0.7.0). Risk: the EWMA log-vol feature is smooth, so
# switches are ~1/yr even at lambda = 5; from lambda = 50 the low-vol state carries the higher mean
# (12.9 % vs 11.2 % ann.) and the 0/1 strategy's Sharpe exceeds buy-and-hold (0.59 vs 0.53) with
# max drawdown -21 % vs -55 %. 50 is also the "typical value" of Shu, Yu & Mulvey (2024).
RISK_PENALTY = 50.0
# Trend: the slow/fast features are persistent themselves, so a light penalty is enough. At
# lambda = 5 the two-feature model beats the raw 252-day sign on the same series (0/1 Sharpe 0.75
# vs 0.60, max drawdown -20 % vs -28 %, 1.9 vs 2.6 switches/yr; on-state mean 15.5 % vs 5.3 % off).
TREND_PENALTY = 5.0
_TOL = 1e-8
_MAX_ITER = 1000


# --------------------------------------------------------------------------- core algorithm
def _loss(Z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """``1/2 ||z_t - mu_k||^2`` for every t, k — the reference convention."""
    d = Z[:, None, :] - centers[None, :, :]
    return 0.5 * np.einsum("tkf,tkf->tk", d, d)


def _viterbi(loss: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Minimum-cost state path (forward values + backtracking). K = 2 uses a scalar recursion on
    the value difference; the general case mirrors the reference ``dp``."""
    n, k = loss.shape
    if k == 2:
        l0, l1 = loss[:, 0].tolist(), loss[:, 1].tolist()
        v0, v1 = l0[0], l1[0]
        back = [0] * n  # bit 0: predecessor of state 0 (0 stay / 1 jump); bit 1: predecessor of state 1
        for t in range(1, n):
            a, b = v1 + lam, v0 + lam  # jump into 0 from 1, jump into 1 from 0
            if v0 <= a:
                n0, p0 = v0, 0
            else:
                n0, p0 = a, 1
            if v1 <= b:
                n1, p1 = v1, 1
            else:
                n1, p1 = b, 0
            v0, v1 = l0[t] + n0, l1[t] + n1
            back[t] = p0 | (p1 << 1)
        labels = np.empty(n, dtype=int)
        s = 0 if v0 <= v1 else 1
        best = v0 if s == 0 else v1
        labels[-1] = s
        for t in range(n - 1, 0, -1):
            s = (back[t] >> s) & 1
            labels[t - 1] = s
        return labels, float(best)
    pen = lam * (1.0 - np.eye(k))
    values = np.empty_like(loss)
    values[0] = loss[0]
    for t in range(1, n):
        values[t] = loss[t] + (values[t - 1][:, None] + pen).min(axis=0)
    labels = np.empty(n, dtype=int)
    labels[-1] = int(values[-1].argmin())
    for t in range(n - 1, 0, -1):
        labels[t - 1] = int((values[t - 1] + pen[:, labels[t]]).argmin())
    return labels, float(values[-1].min())


def _online_last(loss: np.ndarray, lam: float) -> int:
    """State at the last row from the forward values only (the reference ``predict_online``)."""
    n, k = loss.shape
    if k == 2:
        d = float(loss[0, 1] - loss[0, 0])
        for t in range(1, n):
            d = float(loss[t, 1] - loss[t, 0]) + min(max(d, -lam), lam)
        return 0 if d >= 0 else 1
    pen = lam * (1.0 - np.eye(k))
    v = loss[0].copy()
    for t in range(1, n):
        v = loss[t] + (v[:, None] + pen).min(axis=0)
    return int(v.argmin())


def _online_all(loss: np.ndarray, lam: float) -> np.ndarray:
    """Online state for every row (forward values, no backtracking)."""
    n, k = loss.shape
    out = np.empty(n, dtype=int)
    if k == 2:
        d = float(loss[0, 1] - loss[0, 0])
        out[0] = 0 if d >= 0 else 1
        for t in range(1, n):
            d = float(loss[t, 1] - loss[t, 0]) + min(max(d, -lam), lam)
            out[t] = 0 if d >= 0 else 1
        return out
    pen = lam * (1.0 - np.eye(k))
    v = loss[0].copy()
    out[0] = int(v.argmin())
    for t in range(1, n):
        v = loss[t] + (v[:, None] + pen).min(axis=0)
        out[t] = int(v.argmin())
    return out


def _seed(Z: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy k-means++ seeding (Arthur & Vassilvitskii 2007 with the ``2 + log k`` local trials of
    scikit-learn's ``kmeans_plusplus``, the reference implementation's initialiser)."""
    n = len(Z)
    trials = 2 + int(np.log(k))
    centers = [Z[rng.integers(n)]]
    d2 = ((Z - centers[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        tot = d2.sum()
        if tot <= 0:
            cand = rng.integers(n, size=trials)
        else:
            cand = rng.choice(n, size=trials, p=d2 / tot)
        # keep the candidate that lowers the total potential the most
        pots = [np.minimum(d2, ((Z - Z[c]) ** 2).sum(axis=1)).sum() for c in cand]
        best = cand[int(np.argmin(pots))]
        centers.append(Z[best])
        d2 = np.minimum(d2, ((Z - Z[best]) ** 2).sum(axis=1))
    return np.array(centers, dtype=float)


def _quantile_seed(Z: np.ndarray, k: int) -> np.ndarray:
    """Deterministic seed: the rows whose first feature sits at the k equally spaced quantiles."""
    q = np.quantile(Z[:, 0], (np.arange(k) + 0.5) / k)
    idx = [int(np.abs(Z[:, 0] - v).argmin()) for v in q]
    return Z[idx].astype(float)


def _fit(Z: np.ndarray, lam: float, k: int, n_init: int, rng: np.random.Generator) -> np.ndarray:
    """Coordinate descent from a deterministic quantile seed plus ``n_init`` greedy k-means++
    seedings; returns the centres with the lowest objective, ordered by the first feature."""
    best_val, best_centers = np.inf, None
    for i in range(n_init + 1):
        centers = _quantile_seed(Z, k) if i == 0 else _seed(Z, k, rng)
        labels, val = _viterbi(_loss(Z, centers), lam)
        prev_labels, prev_val = None, np.inf
        it = 0
        while it < _MAX_ITER and (prev_labels is None or not np.array_equal(labels, prev_labels)) \
                and prev_val - val > _TOL:
            it += 1
            prev_labels, prev_val = labels, val
            for j in range(k):
                m = labels == j
                if m.any():
                    centers[j] = Z[m].mean(axis=0)
                else:  # empty state: re-seed at the point farthest from the other centres
                    d2 = np.min(((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
                    centers[j] = Z[int(d2.argmax())]
            labels, val = _viterbi(_loss(Z, centers), lam)
        if val < best_val:
            best_val, best_centers = val, centers.copy()
    order = np.argsort(best_centers[:, 0], kind="stable")
    return best_centers[order]


def _standardise(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float | None) -> np.ndarray:
    Z = (X - mu) / sd
    return np.clip(Z, -clip, clip) if clip is not None else Z


def _scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def _periods(index: pd.DatetimeIndex, refit: str) -> pd.PeriodIndex:
    """Period labels for a refit alias given in PARIS's resampling convention (``"ME"``, ``"QE"``,
    ``"YE"``, ``"W"``): pandas periods spell the month/quarter/year aliases without the ``E``."""
    try:
        return index.to_period(refit)
    except ValueError:
        if isinstance(refit, str) and refit.endswith("E") and len(refit) > 1:
            try:
                return index.to_period(refit[:-1])
            except ValueError:
                pass
        raise ValueError(f"refit must be a period alias such as 'ME', 'QE', 'YE' or 'W'; got {refit!r}")


def _features(features: Any) -> pd.DataFrame:
    """Feature frame trimmed to its valid window; interior gaps raise."""
    df = to_frame(features, name="feature")
    df, _, _ = align(df)
    return df


def _check(k: int, lam: float, n_init: int, clip: float | None) -> None:
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError("n_states must be an integer >= 2")
    if not lam >= 0:
        raise ValueError("jump_penalty must be nonnegative")
    if not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError("n_init must be a positive integer")
    if clip is not None and not clip > 0:
        raise ValueError("clip must be positive or None")


# --------------------------------------------------------------------------- public: generic
def jump_labels(features: Any, jump_penalty: float, n_states: int = 2, n_init: int = 10,
                random_state: int = 0, clip: float | None = 3.0) -> pd.Series:
    """In-sample state labels of a jump model fitted to the whole feature history (one Series or a
    DataFrame of features for ONE series). Standardised on the full sample; states ordered by the
    first feature's centre (0 = lowest). This is the hindsight path — use it for research and as
    the ceiling of :func:`jump_states`, never as a signal."""
    _check(n_states, jump_penalty, n_init, clip)
    df = _features(features)
    if len(df) <= n_states:
        raise AlignmentError("too few observations to fit a jump model")
    X = df.to_numpy(dtype=float)
    mu, sd = _scaler(X)
    Z = _standardise(X, mu, sd, clip)
    centers = _fit(Z, jump_penalty, n_states, n_init, np.random.default_rng(random_state))
    labels, _ = _viterbi(_loss(Z, centers), jump_penalty)
    return pd.Series(labels, index=df.index, name="state")


def jump_states(features: Any, jump_penalty: float, window: int = 1260, refit: str | None = "ME",
                n_states: int = 2, lookback: int | None = None, n_init: int = 10, random_state: int = 0,
                clip: float | None = 3.0) -> pd.Series:
    """Causal online states of a rolling jump model (one Series or a DataFrame of features for
    ONE series). At every refit date the scaler and centres are fitted on the ``window``
    observations before it; the state at *t* is the forward-DP argmin over the ``lookback``
    observations ending at *t* (default: ``window``). ``refit`` is a pandas period alias
    (``"ME"`` monthly, ``"QE"`` quarterly, ``"W"`` weekly) or ``None`` for a single fit. The first
    ``window`` observations are NaN. Labels are ordered by the first feature's centre (0 = lowest).
    """
    _check(n_states, jump_penalty, n_init, clip)
    if not isinstance(window, (int, np.integer)) or window < 2 * n_states:
        raise ValueError("window must be an integer >= 2 * n_states")
    lookback = window if lookback is None else lookback
    if not isinstance(lookback, (int, np.integer)) or lookback < 2:
        raise ValueError("lookback must be an integer >= 2")
    df = _features(features)
    n = len(df)
    if window >= n:
        raise AlignmentError(f"window ({window}) must be smaller than the {n} observations")
    X = df.to_numpy(dtype=float)
    if refit is None:
        refit_at = [window]
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("refit needs a DatetimeIndex; pass refit=None for a single fit")
        per = _periods(df.index, refit)
        refit_at = [window] + [i for i in range(window + 1, n) if per[i] != per[i - 1]]
    rng = np.random.default_rng(random_state)
    out = np.full(n, np.nan)
    for j, start in enumerate(refit_at):
        stop = refit_at[j + 1] if j + 1 < len(refit_at) else n
        train = X[start - window:start]
        mu, sd = _scaler(train)
        centers = _fit(_standardise(train, mu, sd, clip), jump_penalty, n_states, n_init, rng)
        for t in range(start, stop):
            lo = max(0, t - lookback + 1)
            Z = _standardise(X[lo:t + 1], mu, sd, clip)
            out[t] = _online_last(_loss(Z, centers), jump_penalty)
    return pd.Series(out, index=df.index, name="state")


# --------------------------------------------------------------------------- features
def _log_vol(r: pd.Series, method: str, lam: float, window: int, warmup: int, ppy: int) -> pd.Series:
    """Log annualised volatility of log returns: EWMA (RiskMetrics, zero-mean) or rolling sd."""
    x = np.log1p(r)
    if method == "ewma":
        if not 0 < lam < 1:
            raise ValueError("vol_lambda must lie in (0, 1)")
        var = (x**2).ewm(alpha=1.0 - lam, adjust=False).mean()
        var.iloc[:warmup] = np.nan
        vol = np.sqrt(var * ppy)
    elif method == "rolling":
        if not isinstance(window, (int, np.integer)) or window < 2:
            raise ValueError("vol_window must be an integer >= 2")
        vol = x.rolling(window).std(ddof=1) * np.sqrt(ppy)
    else:
        raise ValueError("vol must be 'ewma' or 'rolling'")
    with np.errstate(divide="ignore"):
        return np.log(vol.where(vol > 0))


def _default_benchmark(ppy: int) -> pd.Series:
    from paris import regimes

    return regimes._default_benchmark(ppy)


# --------------------------------------------------------------------------- public: risk
def risk_states(returns: Any, basis: str = "own", benchmark: Any = None, vol: str = "ewma",
                vol_lambda: float = 0.94, vol_window: int = 63, vol_warmup: int = 60,
                window: int = 1260, refit: str | None = "ME", jump_penalty: float = RISK_PENALTY,
                lookback: int | None = None, lag: int = 1, periods_per_year: int | None = None,
                n_init: int = 10, random_state: int = 0, clip: float | None = 3.0) -> Any:
    """Risk-on (1) / risk-off (0) from a one-feature jump model on log volatility.

    ``basis="own"`` uses each series' own volatility; ``basis="benchmark"`` uses the benchmark's
    (the bundled S&P 500 proxy at the matching frequency when none is given) and every fund then
    carries the same market state. ``vol="ewma"`` is the RiskMetrics estimator on daily log returns
    (``vol_lambda`` 0.94, first ``vol_warmup`` values NaN); ``vol="rolling"`` the ``vol_window``
    sample sd. The low-volatility state is risk-on. See the module note for the rolling-window
    calibration, online inference and ``lag`` conventions. Series in, Series out; DataFrame in,
    one column per fund; NaN through the warm-up.
    """
    if basis not in ("own", "benchmark"):
        raise ValueError("basis must be 'own' or 'benchmark'")
    if not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError("lag must be a nonnegative integer")
    if basis == "benchmark" and benchmark is None:
        ppy = resolve_periods(prepare(returns, periods_per_year=periods_per_year).returns.index, periods_per_year)
        benchmark = _default_benchmark(ppy)
    p = prepare(returns, benchmark=benchmark if basis == "benchmark" else None, periods_per_year=periods_per_year)

    def states(r: pd.Series) -> pd.Series:
        feat = _log_vol(r, vol, vol_lambda, vol_window, vol_warmup, p.ppy).dropna()
        s = jump_states(feat, jump_penalty, window, refit, 2, lookback, n_init, random_state, clip)
        on = (1.0 - s).reindex(r.index)  # state 0 = low volatility = risk-on
        return on.shift(lag) if lag else on

    if basis == "benchmark":
        market = states(p.benchmark)
        out = pd.DataFrame({c: market for c in p.returns.columns})
    else:
        out = pd.DataFrame({c: states(p.returns[c]) for c in p.returns.columns})
    return out if p.multi else out.iloc[:, 0]


# --------------------------------------------------------------------------- public: trend
def trend_states(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                 slow: int | None = None, fast: int | None = None, compound: bool = False,
                 features: tuple[str, ...] = ("slow", "fast"), feature_weights: Any = None,
                 window: int = 1260, refit: str | None = "ME", jump_penalty: float = TREND_PENALTY,
                 lookback: int | None = None, lag: int = 1, periods_per_year: int | None = None,
                 n_init: int = 10, random_state: int = 0, clip: float | None = 3.0) -> Any:
    """Trend-on (1) / trend-off (0) from a jump model on the momentum turning-point signals.

    ``features`` are the trailing-mean returns of :func:`paris.momentum_signal` — ``("slow",
    "fast")`` by default, ``("slow",)`` for the slow signal alone — on the chosen ``basis`` (raw,
    excess of ``rf`` or relative to ``benchmark``) with the frequency's default lookbacks.
    ``feature_weights`` (same length as ``features``) scale the standardised features before
    clustering; states are ordered by the slow signal's centre and the high state is trend-on.
    Rolling calibration, online inference and ``lag`` as in the module note. Series in, Series
    out; DataFrame in, one column per fund; NaN through the warm-up.
    """
    if not features or any(f not in ("slow", "fast") for f in features) or len(set(features)) != len(features):
        raise ValueError("features must be a non-repeating subset of ('slow', 'fast')")
    if features[0] != "slow":
        raise ValueError("the first feature must be 'slow' (it orders the states)")
    if not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError("lag must be a nonnegative integer")
    w = None
    if feature_weights is not None:
        w = np.asarray(feature_weights, dtype=float).ravel()
        if len(w) != len(features) or (w <= 0).any():
            raise ValueError("feature_weights must be positive and match features")
    p, sig = _signal_frame(returns, basis, rf, benchmark, periods_per_year)
    k_slow, k_fast = _lookbacks(p.ppy, slow, fast)
    trail = {"slow": _trailing(sig, k_slow, compound), "fast": _trailing(sig, k_fast, compound)}

    def states(c: str) -> pd.Series:
        feat = pd.DataFrame({f: trail[f][c] for f in features}).dropna()
        if w is not None:  # weights scale the standardised distances: apply after the scaler
            feat = _weighted_features(feat, w)
        s = jump_states(feat, jump_penalty, window, refit, 2, lookback, n_init, random_state, None if w is not None else clip)
        on = s.reindex(sig.index)  # state 1 = high slow signal = trend-on
        return on.shift(lag) if lag else on

    out = pd.DataFrame({c: states(c) for c in sig.columns})
    return out if p.multi else out.iloc[:, 0]


def _weighted_features(feat: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    """Feature weights multiply the *standardised* features (the reference ``feat_weights``); the
    rolling scaler inside ``jump_states`` re-standardises per window, so weighting is applied to a
    globally standardised, clipped copy — the per-window scaler then only re-centres."""
    X = feat.to_numpy(dtype=float)
    mu, sd = _scaler(X)
    Z = np.clip((X - mu) / sd, -3.0, 3.0) * w
    return pd.DataFrame(Z, index=feat.index, columns=feat.columns)


# --------------------------------------------------------------------------- conditional tables
def _binary_table(states: pd.Series, own: pd.Series, bench: pd.Series | None, ppy: int,
                  labels: tuple[str, str]) -> pd.DataFrame:
    keep = states.notna()
    s, r = states[keep], own[keep]
    b = bench[keep] if bench is not None else None
    rows = {}
    for code, label in enumerate(labels):
        m = (s == code).to_numpy()
        x = r[m]
        row = {"count": float(m.sum()), "frequency": m.mean() if len(s) else float("nan"),
               "own mean (ann.)": float(x.mean() * ppy) if len(x) else float("nan"),
               "own vol (ann.)": float(x.std(ddof=1) * np.sqrt(ppy)) if len(x) > 1 else float("nan")}
        if b is not None:
            y = b[m]
            row["benchmark mean (ann.)"] = float(y.mean() * ppy) if len(y) else float("nan")
            row["benchmark vol (ann.)"] = float(y.std(ddof=1) * np.sqrt(ppy)) if len(y) > 1 else float("nan")
        rows[label] = row
    out = pd.DataFrame(rows).T.reindex(list(labels))
    out.index.name = "state"
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


def risk_state_table(returns: Any, basis: str = "own", benchmark: Any = None, **kwargs: Any) -> pd.DataFrame:
    """Arithmetic average return (annualised), volatility, count and frequency of the periods
    labelled risk-off / risk-on by :func:`risk_states` (same arguments). With ``basis="benchmark"``
    the table also reports the benchmark's own conditional moments. Rows = states; a DataFrame
    input gives one long table with a leading ``fund`` column."""
    ppy_kw = kwargs.get("periods_per_year")
    if basis == "benchmark" and benchmark is None:
        ppy = resolve_periods(prepare(returns, periods_per_year=ppy_kw).returns.index, ppy_kw)
        benchmark = _default_benchmark(ppy)
    p = prepare(returns, benchmark=benchmark if basis == "benchmark" else None, periods_per_year=ppy_kw)
    st = risk_states(p.returns, basis, p.benchmark, **kwargs)
    st = st.to_frame() if isinstance(st, pd.Series) else st
    tables = {c: _binary_table(st[c], p.returns[c], p.benchmark if basis == "benchmark" else None,
                               p.ppy, ("Risk-off", "Risk-on")) for c in p.returns.columns}
    return _long(tables, p.multi)


def trend_state_table(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                      **kwargs: Any) -> pd.DataFrame:
    """Arithmetic average return (annualised), volatility, count and frequency of the periods
    labelled trend-off / trend-on by :func:`trend_states` (same arguments), on the fund's own
    return and, when a benchmark is given or ``basis="relative"``, on the benchmark's. Rows =
    states; a DataFrame input gives one long table with a leading ``fund`` column."""
    ppy_kw = kwargs.get("periods_per_year")
    st = trend_states(returns, basis, rf, benchmark, periods_per_year=ppy_kw,
                      **{k: v for k, v in kwargs.items() if k != "periods_per_year"})
    st = st.to_frame() if isinstance(st, pd.Series) else st
    p, _ = _signal_frame(returns, basis, rf, benchmark, ppy_kw)
    bench = p.benchmark
    if bench is None and benchmark is not None:
        bench = prepare(returns, benchmark=benchmark, periods_per_year=ppy_kw).benchmark.reindex(p.returns.index)
    tables = {c: _binary_table(st[c], p.returns[c], bench, p.ppy, ("Trend-off", "Trend-on"))
              for c in p.returns.columns}
    return _long(tables, p.multi)


# --------------------------------------------------------------------------- combinations
_COMBINE = ("graded", "gate", "and", "or", "cells")


def combine_states(risk: Any, trend: Any, method: str = "graded", cells: dict | None = None) -> Any:
    """Exposure in [0, 1] from the risk-on/off and trend-on/off binaries (aligned; NaN where either
    is NaN). ``"graded"``: ``(risk + trend) / 2``; ``"gate"``: ``trend * (1/2 + risk/2)`` (trend
    gates, risk sizes); ``"and"``: both on; ``"or"``: either on; ``"cells"``: an explicit exposure per
    joint cell, ``cells={(risk, trend): exposure}`` with all four keys — the hook for
    state-conditional sizing estimated elsewhere. Two Series give a Series; two DataFrames with the
    same columns give a DataFrame."""
    if method not in _COMBINE:
        raise ValueError(f"method must be one of {_COMBINE}")
    r, t = risk, trend
    if isinstance(r, pd.DataFrame) != isinstance(t, pd.DataFrame):
        raise AlignmentError("risk and trend must both be Series or both be DataFrames")
    if isinstance(r, pd.DataFrame) and list(r.columns) != list(t.columns):
        raise AlignmentError("risk and trend DataFrames must have the same columns")
    r, t = r.astype(float), t.astype(float).reindex(r.index)
    if method == "cells":
        if cells is None or set(cells) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError("cells needs exposures for the four keys (0,0), (0,1), (1,0), (1,1)")
        for v in cells.values():
            if not 0 <= float(v) <= 1:
                raise ValueError("cell exposures must lie in [0, 1]")
        code = r * 2 + t
        lut = {2 * a + b: float(v) for (a, b), v in cells.items()}
        out = code.replace(lut) if isinstance(code, pd.Series) else code.apply(lambda s: s.replace(lut))
    elif method == "graded":
        out = (r + t) / 2
    elif method == "gate":
        out = t * (0.5 + 0.5 * r)
    elif method == "and":
        out = r * t
    else:
        out = np.maximum(r, t)
    return out.where(r.notna() & t.notna())


def state_table(returns: Any, states: Any, benchmark: Any = None, periods_per_year: int | None = None,
                labels: dict | None = None) -> pd.DataFrame:
    """Count, frequency and the annualised arithmetic mean and volatility of the fund's return (and
    the benchmark's) over the periods carrying each value of ``states`` (any label or code series,
    already aligned in time with ``returns``; ``labels`` renames the values). Rows = state values
    in sorted order; a DataFrame of returns with a DataFrame of states (same columns) gives one
    long table with a leading ``fund`` column."""
    p = prepare(returns, benchmark=benchmark, periods_per_year=periods_per_year)
    st = states.to_frame() if isinstance(states, pd.Series) else states
    if not isinstance(st, pd.DataFrame):
        raise ValueError("states must be a Series or DataFrame")
    if st.shape[1] == 1 and p.returns.shape[1] > 1:
        st = pd.DataFrame({c: st.iloc[:, 0] for c in p.returns.columns})
    if list(st.columns) != list(p.returns.columns):
        st = st.set_axis(p.returns.columns, axis=1) if st.shape[1] == p.returns.shape[1] else st
    if list(st.columns) != list(p.returns.columns):
        raise AlignmentError("states must have one column per fund")

    def one(c: str) -> pd.DataFrame:
        s = st[c].reindex(p.returns.index)
        keep = s.notna()
        s, r = s[keep], p.returns[c][keep]
        b = p.benchmark[keep] if p.benchmark is not None else None
        values = sorted(pd.unique(s))
        rows = {}
        for v in values:
            m = (s == v).to_numpy()
            x = r[m]
            row = {"count": float(m.sum()), "frequency": m.mean(),
                   "own mean (ann.)": float(x.mean() * p.ppy) if len(x) else float("nan"),
                   "own vol (ann.)": float(x.std(ddof=1) * np.sqrt(p.ppy)) if len(x) > 1 else float("nan")}
            if b is not None:
                y = b[m]
                row["benchmark mean (ann.)"] = float(y.mean() * p.ppy) if len(y) else float("nan")
                row["benchmark vol (ann.)"] = float(y.std(ddof=1) * np.sqrt(p.ppy)) if len(y) > 1 else float("nan")
            rows[labels.get(v, v) if labels else v] = row
        out = pd.DataFrame(rows).T
        out.index.name = "state"
        return out

    return _long({c: one(c) for c in p.returns.columns}, p.multi)


_JOINT_LABELS = {0: "Risk-off & Trend-off", 1: "Risk-off & Trend-on", 2: "Risk-on & Trend-off",
                 3: "Risk-on & Trend-on"}


def joint_state_table(returns: Any, benchmark: Any = None, risk_kwargs: dict | None = None,
                      trend_kwargs: dict | None = None, periods_per_year: int | None = None) -> pd.DataFrame:
    """The four joint cells of :func:`risk_states` × :func:`trend_states` (each with its own keyword
    dict; ``window`` etc.), with count, frequency and the annualised mean and volatility of the
    fund's (and the benchmark's) return in each cell — the evidence for how to combine the two."""
    rk, tk = dict(risk_kwargs or {}), dict(trend_kwargs or {})
    p = prepare(returns, benchmark=benchmark, periods_per_year=periods_per_year)
    r = risk_states(p.returns, periods_per_year=p.ppy, **rk)
    t = trend_states(p.returns, periods_per_year=p.ppy, **tk)
    r = r.to_frame() if isinstance(r, pd.Series) else r
    t = t.to_frame() if isinstance(t, pd.Series) else t
    code = (r * 2 + t.reindex(r.index)).where(r.notna() & t.notna())
    if not p.multi:  # a Series input keeps the wide layout (rows = cells), as every other table
        return state_table(p.returns.iloc[:, 0], code.iloc[:, 0], benchmark=p.benchmark,
                           periods_per_year=p.ppy, labels=_JOINT_LABELS)
    return state_table(p.returns, code, benchmark=p.benchmark, periods_per_year=p.ppy, labels=_JOINT_LABELS)


_JOINT_FEATURES = ("logvol", "slow", "fast")


def joint_states(returns: Any, features: tuple[str, ...] = ("logvol", "slow"), n_states: int = 2,
                 feature_weights: Any = None, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                 slow: int | None = None, fast: int | None = None, compound: bool = False,
                 vol: str = "ewma", vol_lambda: float = 0.94, vol_window: int = 63, vol_warmup: int = 60,
                 window: int = 1260, refit: str | None = "ME", jump_penalty: float = TREND_PENALTY,
                 lookback: int | None = None, lag: int = 1, periods_per_year: int | None = None,
                 n_init: int = 10, random_state: int = 0, clip: float | None = 3.0) -> Any:
    """One jump model on several features at once — any ordered subset of ``("logvol", "slow",
    "fast")``: the log volatility of :func:`risk_states` (own returns) and the trailing-mean signals
    of :func:`trend_states` on ``basis``. Returns integer states ``0 .. n_states-1`` ordered by the
    centre of the **first** feature (ascending: with ``"logvol"`` first, 0 is the calmest state;
    with ``"slow"`` first, the highest state is the strongest trend). ``feature_weights`` scale the
    standardised features. Same rolling calibration, online inference and ``lag`` conventions as
    the binary indicators; NaN through the warm-up. Map states to exposure with :func:`state_table`.
    """
    if not features or any(f not in _JOINT_FEATURES for f in features) or len(set(features)) != len(features):
        raise ValueError(f"features must be a non-repeating subset of {_JOINT_FEATURES}")
    if not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError("lag must be a nonnegative integer")
    w = None
    if feature_weights is not None:
        w = np.asarray(feature_weights, dtype=float).ravel()
        if len(w) != len(features) or (w <= 0).any():
            raise ValueError("feature_weights must be positive and match features")
    p, sig = _signal_frame(returns, basis, rf, benchmark, periods_per_year)
    k_slow, k_fast = _lookbacks(p.ppy, slow, fast)
    trail = {"slow": _trailing(sig, k_slow, compound), "fast": _trailing(sig, k_fast, compound)}

    def states(c: str) -> pd.Series:
        cols = {}
        for f in features:
            cols[f] = _log_vol(p.returns[c], vol, vol_lambda, vol_window, vol_warmup, p.ppy) if f == "logvol" else trail[f][c]
        feat = pd.DataFrame(cols).dropna()
        if w is not None:
            feat = _weighted_features(feat, w)
        s = jump_states(feat, jump_penalty, window, refit, n_states, lookback, n_init, random_state,
                        None if w is not None else clip)
        s = s.reindex(p.returns.index)
        return s.shift(lag) if lag else s

    out = pd.DataFrame({c: states(c) for c in p.returns.columns})
    return out if p.multi else out.iloc[:, 0]
