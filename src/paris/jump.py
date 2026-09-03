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
* Calibration is **rolling** by default: at each refit date the model (feature scaler + centres)
  is fitted on the ``window`` observations *before* that date, and the centres stay fixed until the
  next refit (``refit="ME"``: monthly; ``None``: a single fit on the first window).
  ``calibration="expanding"`` fits on *every* observation before the refit date instead, with
  ``window`` as the minimum training length (long memory: the centres reflect the whole history
  seen so far rather than the last ``window`` observations); the online lookback is unchanged.
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
from paris.regimes import (_conditional_table, _lookbacks, _signal_frame, _trailing, _transition_table,
                           momentum_signal)

__all__ = [
    "combine_states",
    "joint_state_table",
    "joint_states",
    "jump_centers",
    "jump_fits",
    "jump_labels",
    "jump_states",
    "risk_centers",
    "risk_fits",
    "risk_signal",
    "risk_state_table",
    "risk_states",
    "state_sizing",
    "state_table",
    "state_transitions",
    "trend_centers",
    "trend_fits",
    "trend_signal",
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
_CALIBRATIONS = ("rolling", "expanding")


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
    rows = loss.tolist()
    values = [rows[0]]
    for t in range(1, n):
        v = values[-1]
        m = min(v) + lam
        row = rows[t]
        values.append([row[i] + (v[i] if v[i] <= m else m) for i in range(k)])
    labels = np.empty(n, dtype=int)
    labels[-1] = int(np.argmin(values[-1]))
    for t in range(n - 1, 0, -1):
        j = labels[t]
        v = values[t - 1]
        labels[t - 1] = int(np.argmin([v[i] + (0.0 if i == j else lam) for i in range(k)]))
    return labels, float(min(values[-1]))


def _online_last(loss: np.ndarray, lam: float) -> int:
    """State at the last row from the forward values only (the reference ``predict_online``)."""
    n, k = loss.shape
    if k == 2:
        d = float(loss[0, 1] - loss[0, 0])
        for t in range(1, n):
            d = float(loss[t, 1] - loss[t, 0]) + min(max(d, -lam), lam)
        return 0 if d >= 0 else 1
    # general K: min_j (v_j + lam * [j != i]) = min(v_i, min_j v_j + lam); plain lists beat numpy
    # on 2x2..4x4 problems by an order of magnitude and give identical numbers
    rows = loss.tolist()
    v = rows[0]
    for t in range(1, n):
        m = min(v) + lam
        row = rows[t]
        v = [row[i] + (v[i] if v[i] <= m else m) for i in range(k)]
    return int(np.argmin(v))


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
    rows = loss.tolist()
    v = rows[0]
    out[0] = int(np.argmin(v))
    for t in range(1, n):
        m = min(v) + lam
        row = rows[t]
        v = [row[i] + (v[i] if v[i] <= m else m) for i in range(k)]
        out[t] = int(np.argmin(v))
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


def _rolling_fits(features, jump_penalty, window, refit, n_states, lookback, n_init, random_state, clip,
                  feature_weights, calibration="rolling", fits=None):
    """Validate the inputs and fit the model at every refit date on the ``window`` observations
    before it (``calibration="rolling"``) or on all observations before it (``"expanding"``,
    ``window`` = minimum training length). Returns ``(frame, X, weights, lookback, [(start, mu, sd, centres), ...])``; the
    centres are in standardised (and weighted) units, ordered by the first feature."""
    _check(n_states, jump_penalty, n_init, clip)
    if calibration not in _CALIBRATIONS:
        raise ValueError(f"calibration must be one of {_CALIBRATIONS}")
    w = None
    if feature_weights is not None:
        w = np.asarray(feature_weights, dtype=float).ravel()
        if (w <= 0).any():
            raise ValueError("feature_weights must be positive")
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
    if w is not None and len(w) != X.shape[1]:
        raise ValueError("feature_weights must match the number of features")
    if refit is None:
        refit_at = [window]
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("refit needs a DatetimeIndex; pass refit=None for a single fit")
        per = _periods(df.index, refit)
        refit_at = [window] + [i for i in range(window + 1, n) if per[i] != per[i - 1]]
    prior = _unpack_fits(fits, df, n_states) if fits is not None else {}
    out = []
    for start in refit_at:
        key = df.index[start]
        if key in prior:
            out.append((start, *prior[key]))
            continue
        train = X[start - window:start] if calibration == "rolling" else X[:start]
        mu, sd = _scaler(train)
        Zt = _standardise(train, mu, sd, clip)
        # one generator per refit date, seeded by (random_state, refit position): a run resumed from
        # stored fits reproduces a cold run exactly
        rng = np.random.default_rng([int(random_state), int(start)])
        out.append((start, mu, sd, _fit(Zt if w is None else Zt * w, jump_penalty, n_states, n_init, rng)))
    return df, X, w, int(lookback), out


def _pack_fits(df: pd.DataFrame, fits: list, n_states: int) -> pd.DataFrame:
    rows = {}
    for start, mu, sd, centers in fits:
        row = {}
        for i, f in enumerate(df.columns):
            row[f"mu {f}"] = float(mu[i])
            row[f"sd {f}"] = float(sd[i])
            for k in range(n_states):
                row[f"state{k} {f}"] = float(centers[k, i])
        rows[df.index[start]] = row
    out = pd.DataFrame(rows).T
    out.index.name = "refit"
    return out


def _unpack_fits(fits: pd.DataFrame, df: pd.DataFrame, n_states: int) -> dict:
    if not isinstance(fits, pd.DataFrame):
        raise ValueError("fits must be the DataFrame returned by jump_fits")
    p = len(df.columns)
    need = [f"{pre} {f}" for f in df.columns for pre in ("mu", "sd")] + [f"state{k} {f}" for k in range(n_states) for f in df.columns]
    missing = [c for c in need if c not in fits.columns]
    extra = [c for c in fits.columns if c.startswith("state") and int(c.split(" ", 1)[0][5:]) >= n_states]
    if missing or extra:
        raise ValueError(f"fits do not match the features / n_states: missing {missing[:3]}, extra {extra[:3]}")
    out = {}
    for ts, row in fits.iterrows():
        mu = np.array([row[f"mu {f}"] for f in df.columns], dtype=float)
        sd = np.array([row[f"sd {f}"] for f in df.columns], dtype=float)
        centers = np.array([[row[f"state{k} {f}"] for f in df.columns] for k in range(n_states)], dtype=float).reshape(n_states, p)
        out[pd.Timestamp(ts)] = (mu, sd, centers)
    return out


def jump_centers(features: Any, jump_penalty: float, window: int = 1260, refit: str | None = "ME",
                 n_states: int = 2, n_init: int = 10, random_state: int = 0, clip: float | None = 3.0,
                 feature_weights: Any = None, calibration: str = "rolling") -> pd.DataFrame:
    """What the rolling model fitted: one row per refit date (the first date the fit applies to)
    with the cluster centres in the **original feature units** (``state<k> <feature>`` columns,
    states ordered by the first feature) and, for a one-feature two-state model, ``threshold``: the
    midpoint of the two centres, the zero-penalty switching level (the jump penalty moves the
    effective switch beyond it by an amount that grows with the penalty). Same arguments and
    calibration as :func:`jump_states`."""
    df, X, w, _, fits = _rolling_fits(features, jump_penalty, window, refit, n_states, None, n_init,
                                      random_state, clip, feature_weights, calibration)
    rows = {}
    for start, mu, sd, centers in fits:
        c = centers / w if w is not None else centers
        orig = c * sd + mu  # (K, p) in original units
        row = {f"state{k} {f}": float(orig[k, i]) for k in range(n_states) for i, f in enumerate(df.columns)}
        if n_states == 2 and X.shape[1] == 1:
            row["threshold"] = float(orig[:, 0].mean())
        rows[df.index[start]] = row
    out = pd.DataFrame(rows).T
    out.index.name = "refit"
    return out


def jump_fits(features: Any, jump_penalty: float, window: int = 1260, refit: str | None = "ME",
              n_states: int = 2, n_init: int = 10, random_state: int = 0, clip: float | None = 3.0,
              feature_weights: Any = None, calibration: str = "rolling", fits: pd.DataFrame | None = None) -> pd.DataFrame:
    """The fitted model at every refit date, in a form that can be stored and resumed: one row per
    refit (indexed by the first date the fit applies to) with the training scaler (``mu <feature>``,
    ``sd <feature>``) and the centres in standardised, weighted units (``state<k> <feature>``).
    Pass the result back as ``fits`` to :func:`jump_states` or :func:`jump_fits` on a longer history
    with the SAME arguments: refits already present are reused and only new refit dates are fitted,
    and the result is identical to a cold run (each refit is seeded by ``(random_state, position)``).
    Mismatched arguments are the caller's responsibility; mismatched features or ``n_states``
    raise ``ValueError``."""
    df, _, _, _, out = _rolling_fits(features, jump_penalty, window, refit, n_states, None, n_init,
                                     random_state, clip, feature_weights, calibration, fits)
    return _pack_fits(df, out, n_states)


def jump_states(features: Any, jump_penalty: float, window: int = 1260, refit: str | None = "ME",
                n_states: int = 2, lookback: int | None = None, n_init: int = 10, random_state: int = 0,
                clip: float | None = 3.0, feature_weights: Any = None, calibration: str = "rolling",
                fits: pd.DataFrame | None = None, since: Any = None) -> pd.Series:
    """Causal online states of a rolling jump model (one Series or a DataFrame of features for
    ONE series). At every refit date the scaler and centres are fitted on the ``window``
    observations before it; the state at *t* is the forward-DP argmin over the ``lookback``
    observations ending at *t* (default: ``window``). ``refit`` is a pandas period alias
    (``"ME"`` monthly, ``"QE"`` quarterly, ``"W"`` weekly) or ``None`` for a single fit. The first
    ``window`` observations are NaN. Labels are ordered by the first feature's centre (0 = lowest).
    ``feature_weights`` multiply the standardised, clipped features (the reference ``feat_weights``)
    inside every training and lookback window, so they scale the distances without touching the
    per-window scaler and without any full-sample statistic. ``calibration="expanding"`` fits each
    refit on all observations before it (``window`` = minimum), see the module note. ``fits`` (from
    :func:`jump_fits` on an earlier history, same arguments) resumes: stored refits are reused, new
    ones fitted, and only the online inference runs over the whole history. ``since`` (a date)
    restricts the online inference to observations at or after it — the state at *t* depends only
    on the fit in force and the ``lookback`` observations ending at *t*, so the values returned are
    identical to a full run; earlier observations are NaN.
    """
    df, X, w, lookback, fl = _rolling_fits(features, jump_penalty, window, refit, n_states, lookback, n_init,
                                           random_state, clip, feature_weights, calibration, fits)
    n = len(df)
    out = np.full(n, np.nan)
    first_t = 0 if since is None else int(df.index.searchsorted(pd.Timestamp(since)))
    for j, (start, mu, sd, centers) in enumerate(fl):
        stop = fl[j + 1][0] if j + 1 < len(fl) else n
        for t in range(max(start, first_t), stop):
            lo = max(0, t - lookback + 1)
            Z = _standardise(X[lo:t + 1], mu, sd, clip)
            out[t] = _online_last(_loss(Z if w is None else Z * w, centers), jump_penalty)
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


def _since_feature(feat_index: pd.Index, since: Any, lag: int) -> Any:
    """``since`` refers to the indicator's dates (after ``lag``); the online inference must start
    ``lag`` observations earlier on the feature index."""
    if since is None:
        return None
    pos = int(feat_index.searchsorted(pd.Timestamp(since)))
    return feat_index[max(pos - lag, 0)]


def _default_benchmark(ppy: int) -> pd.Series:
    from paris import regimes

    return regimes._default_benchmark(ppy)


# --------------------------------------------------------------------------- public: risk
def risk_states(returns: Any, basis: str = "own", benchmark: Any = None, vol: str = "ewma",
                vol_lambda: float = 0.94, vol_window: int = 63, vol_warmup: int = 60,
                window: int = 1260, refit: str | None = "ME", jump_penalty: float = RISK_PENALTY,
                lookback: int | None = None, lag: int = 1, periods_per_year: int | None = None,
                n_init: int = 10, random_state: int = 0, clip: float | None = 3.0,
                calibration: str = "rolling", fits: pd.DataFrame | None = None, since: Any = None) -> Any:
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
        feat = _log_vol(r, vol, vol_lambda, vol_window, vol_warmup, p.ppy).dropna().rename("vol")
        s = jump_states(feat, jump_penalty, window, refit, 2, lookback, n_init, random_state, clip,
                        calibration=calibration, fits=fits, since=_since_feature(feat.index, since, lag))
        on = (1.0 - s).reindex(r.index)  # state 0 = low volatility = risk-on
        return on.shift(lag) if lag else on

    if basis == "benchmark":
        market = states(p.benchmark)
        out = pd.DataFrame({c: market for c in p.returns.columns})
    else:
        out = pd.DataFrame({c: states(p.returns[c]) for c in p.returns.columns})
    return out if p.multi else out.iloc[:, 0]


def risk_signal(returns: Any, vol: str = "ewma", vol_lambda: float = 0.94, vol_window: int = 63,
                vol_warmup: int = 60, periods_per_year: int | None = None, log: bool = False) -> Any:
    """The volatility series :func:`risk_states` classifies: annualised EWMA (RiskMetrics,
    ``vol_lambda``) or rolling (``vol_window``) volatility of daily log returns, NaN through
    ``vol_warmup``; ``log=True`` returns its natural log, the model's actual feature. Series in,
    Series out; DataFrame in, one column per fund."""
    p = prepare(returns, periods_per_year=periods_per_year)
    out = pd.DataFrame({c: _log_vol(p.returns[c], vol, vol_lambda, vol_window, vol_warmup, p.ppy) for c in p.returns.columns})
    if not log:
        out = np.exp(out)
    return out if p.multi else out.iloc[:, 0]


def trend_signal(returns: Any, signal: str = "slow", basis: str = "raw", rf: Any = None, benchmark: Any = None,
                 periods_per_year: int | None = None, slow: int | None = None, fast: int | None = None,
                 compound: bool = False) -> Any:
    """The trailing-mean return series :func:`trend_states` classifies — the ``"slow"`` or
    ``"fast"`` signal of :func:`paris.momentum_signal` with the same basis and lookbacks."""
    return momentum_signal(returns, signal, basis, rf, benchmark, periods_per_year, slow, fast, compound)


def risk_fits(returns: Any, vol: str = "ewma", vol_lambda: float = 0.94, vol_window: int = 63, vol_warmup: int = 60,
              window: int = 1260, refit: str | None = "ME", jump_penalty: float = RISK_PENALTY,
              periods_per_year: int | None = None, n_init: int = 10, random_state: int = 0, clip: float | None = 3.0,
              calibration: str = "rolling", fits: pd.DataFrame | None = None) -> pd.DataFrame:
    """:func:`jump_fits` of the risk model on ONE series (the stored form of its calibration); pass the
    result as ``fits`` to :func:`risk_states` on a longer history with the same arguments."""
    p = prepare(returns, periods_per_year=periods_per_year)
    if p.multi:
        raise ValueError("risk_fits takes one series")
    feat = _log_vol(p.returns.iloc[:, 0], vol, vol_lambda, vol_window, vol_warmup, p.ppy).dropna().rename("vol")
    return jump_fits(feat, jump_penalty, window, refit, 2, n_init, random_state, clip, None, calibration, fits)


def trend_fits(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None, slow: int | None = None,
               fast: int | None = None, compound: bool = False, features: tuple[str, ...] = ("slow", "fast"),
               feature_weights: Any = None, window: int = 1260, refit: str | None = "ME",
               jump_penalty: float = TREND_PENALTY, periods_per_year: int | None = None, n_init: int = 10,
               random_state: int = 0, clip: float | None = 3.0, calibration: str = "rolling",
               fits: pd.DataFrame | None = None) -> pd.DataFrame:
    """:func:`jump_fits` of the trend model on ONE series; pass the result as ``fits`` to
    :func:`trend_states` on a longer history with the same arguments."""
    p, sig = _signal_frame(returns, basis, rf, benchmark, periods_per_year)
    if p.multi:
        raise ValueError("trend_fits takes one series")
    k_slow, k_fast = _lookbacks(p.ppy, slow, fast)
    trail = {"slow": _trailing(sig, k_slow, compound), "fast": _trailing(sig, k_fast, compound)}
    c = sig.columns[0]
    feat = pd.DataFrame({f: trail[f][c] for f in features}).dropna()
    return jump_fits(feat, jump_penalty, window, refit, 2, n_init, random_state, clip, feature_weights, calibration, fits)


def _centers_long(tables: dict[str, pd.DataFrame], multi: bool) -> pd.DataFrame:
    if not multi:
        return next(iter(tables.values()))
    parts = []
    for name, tb in tables.items():
        tb = tb.reset_index()
        tb.insert(0, "fund", name)
        parts.append(tb)
    return pd.concat(parts, ignore_index=True)


def risk_centers(returns: Any, vol: str = "ewma", vol_lambda: float = 0.94, vol_window: int = 63,
                 vol_warmup: int = 60, window: int = 1260, refit: str | None = "ME",
                 jump_penalty: float = RISK_PENALTY, periods_per_year: int | None = None, n_init: int = 10,
                 random_state: int = 0, clip: float | None = 3.0, log: bool = False,
                 calibration: str = "rolling") -> pd.DataFrame:
    """The fitted risk model per refit date, in annualised-volatility units (``log=True``: in the
    log units the model sees): ``state0 vol`` (the risk-on centre), ``state1 vol`` (risk-off) and
    ``threshold`` (their midpoint) — read as "risk-off above about x % annualised". Own-volatility
    basis; a DataFrame input gives one long table with a leading ``fund`` column."""
    p = prepare(returns, periods_per_year=periods_per_year)
    tabs = {}
    for c in p.returns.columns:
        feat = _log_vol(p.returns[c], vol, vol_lambda, vol_window, vol_warmup, p.ppy).dropna().rename("vol")
        tb = jump_centers(feat, jump_penalty, window, refit, 2, n_init, random_state, clip, calibration=calibration)
        tabs[c] = tb if log else np.exp(tb)
    return _centers_long(tabs, p.multi)


def trend_centers(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                  slow: int | None = None, fast: int | None = None, compound: bool = False,
                  features: tuple[str, ...] = ("slow", "fast"), feature_weights: Any = None,
                  window: int = 1260, refit: str | None = "ME", jump_penalty: float = TREND_PENALTY,
                  periods_per_year: int | None = None, n_init: int = 10, random_state: int = 0,
                  clip: float | None = 3.0, calibration: str = "rolling") -> pd.DataFrame:
    """The fitted trend model per refit date in signal units (trailing-mean period returns):
    ``state0 slow`` / ``state1 slow`` (trend-off / trend-on centres), the same for ``fast`` when
    used, and ``threshold`` for the slow-only model. A DataFrame input gives one long table with a
    leading ``fund`` column."""
    if not features or any(f not in ("slow", "fast") for f in features) or len(set(features)) != len(features) or features[0] != "slow":
        raise ValueError("features must be ('slow',) or ('slow', 'fast')")
    p, sig = _signal_frame(returns, basis, rf, benchmark, periods_per_year)
    k_slow, k_fast = _lookbacks(p.ppy, slow, fast)
    trail = {"slow": _trailing(sig, k_slow, compound), "fast": _trailing(sig, k_fast, compound)}
    tabs = {}
    for c in sig.columns:
        feat = pd.DataFrame({f: trail[f][c] for f in features}).dropna()
        tabs[c] = jump_centers(feat, jump_penalty, window, refit, 2, n_init, random_state, clip, feature_weights,
                               calibration)
    return _centers_long(tabs, p.multi)


# --------------------------------------------------------------------------- public: trend
def trend_states(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                 slow: int | None = None, fast: int | None = None, compound: bool = False,
                 features: tuple[str, ...] = ("slow", "fast"), feature_weights: Any = None,
                 window: int = 1260, refit: str | None = "ME", jump_penalty: float = TREND_PENALTY,
                 lookback: int | None = None, lag: int = 1, periods_per_year: int | None = None,
                 n_init: int = 10, random_state: int = 0, clip: float | None = 3.0,
                 calibration: str = "rolling", fits: pd.DataFrame | None = None, since: Any = None) -> Any:
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
        s = jump_states(feat, jump_penalty, window, refit, 2, lookback, n_init, random_state, clip, w, calibration, fits,
                        _since_feature(feat.index, since, lag))
        on = s.reindex(sig.index)  # state 1 = high slow signal = trend-on
        return on.shift(lag) if lag else on

    out = pd.DataFrame({c: states(c) for c in sig.columns})
    return out if p.multi else out.iloc[:, 0]


# --------------------------------------------------------------------------- conditional tables
def _long(tables: dict[str, pd.DataFrame], multi: bool) -> pd.DataFrame:
    if not multi:
        return next(iter(tables.values()))
    parts = []
    for name, tb in tables.items():
        tb = tb.reset_index()
        tb.insert(0, "fund", name)
        parts.append(tb)
    return pd.concat(parts, ignore_index=True)


_RISK_LABELS = {0.0: "Risk-off", 1.0: "Risk-on"}
_TREND_LABELS = {0.0: "Trend-off", 1.0: "Trend-on"}


def risk_state_table(returns: Any, basis: str = "own", benchmark: Any = None, rf: Any = None,
                     **kwargs: Any) -> pd.DataFrame:
    """The unified state table (see :func:`state_table`) for the periods labelled risk-off /
    risk-on by :func:`risk_states` (same arguments): count, frequency, the fund's mean, vol,
    skewness and up-frequency, and — with ``basis="benchmark"`` or a ``benchmark`` — the
    benchmark's mean and vol and the active mean; ``rf`` adds the excess means. Rows = states; a
    DataFrame input gives one long table with a leading ``fund`` column."""
    ppy_kw = kwargs.get("periods_per_year")
    if basis == "benchmark" and benchmark is None:
        ppy = resolve_periods(prepare(returns, periods_per_year=ppy_kw).returns.index, ppy_kw)
        benchmark = _default_benchmark(ppy)
    p = prepare(returns, benchmark=benchmark, rf=rf if rf is not None else 0.0, periods_per_year=ppy_kw)
    st = risk_states(p.returns, basis, p.benchmark, **kwargs)
    own = p.returns if p.multi else p.returns.iloc[:, 0]
    return state_table(own, st, benchmark=p.benchmark, rf=rf, periods_per_year=p.ppy, labels=_RISK_LABELS)


def trend_state_table(returns: Any, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                      **kwargs: Any) -> pd.DataFrame:
    """The unified state table for the periods labelled trend-off / trend-on by
    :func:`trend_states` (same arguments): the fund's own moments and, when a benchmark is given
    or ``basis="relative"``, the benchmark's mean and vol and the active mean; ``rf`` adds the
    excess means. Rows = states; a DataFrame input gives one long table with a leading ``fund``
    column."""
    ppy_kw = kwargs.get("periods_per_year")
    st = trend_states(returns, basis, rf, benchmark, periods_per_year=ppy_kw,
                      **{k: v for k, v in kwargs.items() if k != "periods_per_year"})
    p, _ = _signal_frame(returns, basis, rf, benchmark, ppy_kw)
    bench = p.benchmark
    if bench is None and benchmark is not None:
        bench = prepare(returns, benchmark=benchmark, periods_per_year=ppy_kw).benchmark
    own = p.returns if p.multi else p.returns.iloc[:, 0]
    return state_table(own, st, benchmark=bench, rf=rf, periods_per_year=p.ppy, labels=_TREND_LABELS)


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


def _states_frame_like(states: Any, returns: pd.DataFrame) -> pd.DataFrame:
    st = states.to_frame() if isinstance(states, pd.Series) else states
    if not isinstance(st, pd.DataFrame):
        raise ValueError("states must be a Series or DataFrame")
    if st.shape[1] == 1 and returns.shape[1] > 1:
        st = pd.DataFrame({c: st.iloc[:, 0] for c in returns.columns})
    if list(st.columns) != list(returns.columns):
        if st.shape[1] == returns.shape[1]:
            st = st.set_axis(returns.columns, axis=1)
        else:
            raise AlignmentError("states must have one column per fund")
    return st


def state_table(returns: Any, states: Any, benchmark: Any = None, rf: Any = None,
                periods_per_year: int | None = None, labels: dict | None = None, shift: int = 0) -> pd.DataFrame:
    """The unified state-conditional table for ANY label or code series aligned in time with
    ``returns``: count, frequency, the fund's annualised mean, volatility, population skewness and
    up-frequency, the benchmark's mean and vol and the active (fund minus benchmark) mean when a
    benchmark is given, and the excess means when ``rf`` is given. ``shift=0`` pairs the label at
    *T* with the return of *T* (the convention for the already-lagged jump-model labels);
    ``shift=1`` pairs the label at *t* with the return at *t+1* (the momentum-state convention).
    ``labels`` renames the values and fixes the row order. Rows = state values; a DataFrame of
    returns (with a matching DataFrame or one broadcast Series of states) gives one long table with
    a leading ``fund`` column."""
    if not isinstance(shift, (int, np.integer)) or shift < 0:
        raise ValueError("shift must be a nonnegative integer")
    p = prepare(returns, benchmark=benchmark, rf=rf if rf is not None else 0.0, periods_per_year=periods_per_year)
    st = _states_frame_like(states, p.returns)
    rf_s = p.rf if rf is not None else None
    tables = {c: _conditional_table(st[c].reindex(p.returns.index), p.returns[c], p.benchmark, rf_s, p.ppy,
                                    labels, int(shift)) for c in p.returns.columns}
    return _long(tables, p.multi)


def state_transitions(states: Any, order: list | None = None) -> pd.DataFrame:
    """Transition probabilities from the value at *t* (rows) to the value at *t+1* (columns) of any
    label or code series (momentum labels, risk / trend binaries, joint codes); ``order`` fixes the
    row and column order. A DataFrame gives one long table with a leading ``fund`` column."""
    st = states.to_frame() if isinstance(states, pd.Series) else states
    if not isinstance(st, pd.DataFrame):
        raise ValueError("states must be a Series or DataFrame")
    return _long({c: _transition_table(st[c], order) for c in st.columns}, not isinstance(states, pd.Series))


def state_sizing(returns: Any, states: Any, rf: Any = 0.0, window: int | None = None, refit: str | None = "YE",
                 min_obs: int = 20, periods_per_year: int | None = None, table: bool = False) -> Any:
    """Causal state-conditional exposure: at every refit date the per-state Sharpe ratio of the
    fund's excess return is estimated on the history before it (expanding, or the trailing
    ``window`` observations) and each state's exposure is ``clip(SR_k / max_k SR_k, 0, 1)`` (0 for
    a state with fewer than ``min_obs`` observations or when no state has a positive Sharpe); the
    mapping is applied to the labels until the next refit. ``states`` must already be causal (the
    jump-model labels are). Returns the exposure series (NaN before the first refit), or with
    ``table=True`` the per-refit exposures per state. Series in, Series out; DataFrame in, one
    column per fund (long table with ``table=True``)."""
    if not isinstance(min_obs, (int, np.integer)) or min_obs < 2:
        raise ValueError("min_obs must be an integer >= 2")
    if window is not None and (not isinstance(window, (int, np.integer)) or window < min_obs):
        raise ValueError("window must be None or an integer >= min_obs")
    p = prepare(returns, rf=rf, periods_per_year=periods_per_year)
    st = _states_frame_like(states, p.returns)
    idx = p.returns.index
    if refit is not None:
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("refit needs a DatetimeIndex; pass refit=None for a single estimate")
        per = _periods(idx, refit)

    def one(c: str) -> tuple[pd.Series, pd.DataFrame]:
        s = st[c].reindex(idx).to_numpy(dtype=float)
        ex = p.excess[c].to_numpy(dtype=float)
        n = len(s)
        valid = ~np.isnan(s)
        first = int(np.argmax(valid)) if valid.any() else n
        # refit positions: first observation of each period after enough labelled history
        cand = [i for i in range(first + 1, n) if refit is None or per[i] != per[i - 1]]
        starts = [i for i in cand if np.sum(valid[max(0, i - window) if window else 0:i]) >= min_obs]
        if refit is None:
            starts = starts[:1]
        expo = np.full(n, np.nan)
        rows = {}
        values = sorted(pd.unique(s[valid]))
        for j, i in enumerate(starts):
            stop = starts[j + 1] if j + 1 < len(starts) else n
            lo = max(0, i - window) if window else 0
            hs, hx = s[lo:i], ex[lo:i]
            sr = {}
            for v in values:
                m = hs == v
                x = hx[m]
                sr[v] = float(x.mean() / x.std(ddof=1)) if m.sum() >= min_obs and x.std(ddof=1) > 0 else 0.0
            top = max(sr.values()) if sr else 0.0
            mapping = {v: (float(np.clip(q / top, 0.0, 1.0)) if top > 0 else 0.0) for v, q in sr.items()}
            rows[idx[i]] = mapping
            seg = s[i:stop]
            expo[i:stop] = [mapping.get(v, np.nan) if not np.isnan(v) else np.nan for v in seg]
        tb = pd.DataFrame(rows).T.reindex(columns=values)
        tb.index.name = "refit"
        return pd.Series(expo, index=idx, name="exposure"), tb

    res = {c: one(c) for c in p.returns.columns}
    if table:
        return _long({c: v[1] for c, v in res.items()}, p.multi)
    out = pd.DataFrame({c: v[0] for c, v in res.items()})
    return out if p.multi else out.iloc[:, 0]


_JOINT_LABELS = {0: "Risk-off & Trend-off", 1: "Risk-off & Trend-on", 2: "Risk-on & Trend-off",
                 3: "Risk-on & Trend-on"}
_JOINT_LABELS_F = {float(k): v for k, v in _JOINT_LABELS.items()}


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
                           periods_per_year=p.ppy, labels=_JOINT_LABELS_F)
    return state_table(p.returns, code, benchmark=p.benchmark, periods_per_year=p.ppy, labels=_JOINT_LABELS_F)


_JOINT_FEATURES = ("logvol", "slow", "fast")


def joint_states(returns: Any, features: tuple[str, ...] = ("logvol", "slow"), n_states: int = 2,
                 feature_weights: Any = None, basis: str = "raw", rf: Any = None, benchmark: Any = None,
                 slow: int | None = None, fast: int | None = None, compound: bool = False,
                 vol: str = "ewma", vol_lambda: float = 0.94, vol_window: int = 63, vol_warmup: int = 60,
                 window: int = 1260, refit: str | None = "ME", jump_penalty: float = TREND_PENALTY,
                 lookback: int | None = None, lag: int = 1, periods_per_year: int | None = None,
                 n_init: int = 10, random_state: int = 0, clip: float | None = 3.0,
                 calibration: str = "rolling") -> Any:
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
        s = jump_states(feat, jump_penalty, window, refit, n_states, lookback, n_init, random_state, clip, w, calibration)
        s = s.reindex(p.returns.index)
        return s.shift(lag) if lag else s

    out = pd.DataFrame({c: states(c) for c in p.returns.columns})
    return out if p.multi else out.iloc[:, 0]
