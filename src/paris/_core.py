"""Shared plumbing for PARIS: input coercion, alignment, frequency, risk-free rate, annualisation.

This is the ONLY module other paris modules import from. Everything here is convention-free
helper code; the statistical conventions live in the topic modules.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "AlignmentError",
    "FrequencyError",
    "GapError",
    "ParisError",
    "Prepared",
    "align",
    "annualize_return",
    "annualize_vol",
    "apply_cols",
    "infer_periods",
    "is_multi",
    "prepare",
    "resolve_periods",
    "rf_annual",
    "rf_per_period",
    "safe_div",
    "to_frame",
    "to_series",
    "to_weights",
]

# Day-spacing (median) -> periods per year.  Tolerant to weekends/holidays/month-length.
_FREQ_BANDS: tuple[tuple[float, float, int], ...] = (
    (0.5, 3.5, 252),
    (6.0, 8.0, 52),
    (27.0, 32.0, 12),
    (85.0, 95.0, 4),
    (360.0, 371.0, 1),
)


class ParisError(ValueError):
    """Base class for PARIS input errors."""


class GapError(ParisError):
    """Interior missing observations inside the common window (never filled, never dropped)."""


class FrequencyError(ParisError):
    """Periods per year could not be inferred and none was supplied."""


class AlignmentError(ParisError):
    """Index is unsorted, duplicated, or series share no common window."""


# --------------------------------------------------------------------------- coercion
def is_multi(x: Any) -> bool:
    """True if *x* represents several return series (DataFrame / 2-D array / multi-col polars)."""
    if isinstance(x, pd.DataFrame):
        return True
    if isinstance(x, pd.Series):
        return False
    if hasattr(x, "to_pandas"):  # polars
        return hasattr(x, "columns") and len([c for c in x.columns if x[c].dtype.is_numeric()]) > 1
    return np.ndim(x) == 2


def _from_polars(x: Any) -> pd.DataFrame | pd.Series:
    obj = x.to_pandas()
    if isinstance(obj, pd.DataFrame):
        dt_cols = [c for c in obj.columns if np.issubdtype(obj[c].dtype, np.datetime64)]
        if dt_cols:
            obj = obj.set_index(dt_cols[0])
            obj.index.name = None
    return obj


def to_frame(x: Any, name: str = "returns") -> pd.DataFrame:
    """Coerce Series / DataFrame / ndarray / list / polars into a float DataFrame, validating the index."""
    if hasattr(x, "to_pandas"):
        x = _from_polars(x)
    if isinstance(x, pd.DataFrame):
        df = x.copy()
    elif isinstance(x, pd.Series):
        df = x.to_frame(name=x.name if x.name is not None else name)
    else:
        arr = np.asarray(x)
        if arr.ndim == 1:
            df = pd.DataFrame({name: arr})
        elif arr.ndim == 2:
            df = pd.DataFrame(arr, columns=[f"{name}_{i}" for i in range(arr.shape[1])])
        else:  # pragma: no cover - defensive
            raise TypeError("returns must be 1-D or 2-D")
    try:
        df = df.astype(float)
    except (TypeError, ValueError) as e:
        raise ParisError("returns must be numeric") from e
    if not df.index.is_monotonic_increasing:
        raise AlignmentError("index must be sorted ascending")
    if df.index.has_duplicates:
        raise AlignmentError("index has duplicate labels")
    return df


def to_series(x: Any, name: str = "benchmark") -> pd.Series:
    """Coerce a single return series (benchmark / rf) to a float Series."""
    df = to_frame(x, name=name)
    if df.shape[1] != 1:
        raise AlignmentError(f"expected a single series, got {df.shape[1]} columns")
    return df.iloc[:, 0]


# --------------------------------------------------------------------------- weights
WEIGHT_TOL = 1e-4  # a weight row must sum to 1 within this (four decimals) unless normalised


def to_weights(weights: Any, columns: Any, index: pd.Index | None = None, *,
               normalize: bool = False, tol: float = WEIGHT_TOL) -> pd.DataFrame:
    """Coerce portfolio weights to a DataFrame (rows = weight dates, columns = assets).

    A one-time vector (list / array positional in *columns* order; dict / Series by asset name)
    becomes a single row dated one day before ``index[0]`` when *index* is given — i.e. "before the
    first return", so it applies to every return — else at position 0. A DataFrame keeps its dates.
    Assets named in the weights must exist in *columns* (``AlignmentError``); columns of *columns*
    without a weight are dropped. NaN
    weights raise ``ParisError``. Each row must sum to 1 within *tol* (``ParisError``), unless
    ``normalize=True`` divides every row by its sum (a zero sum cannot be normalised).
    """
    cols = list(columns)
    if hasattr(weights, "to_pandas"):
        weights = _from_polars(weights)
    if isinstance(weights, pd.DataFrame):
        df = to_frame(weights, name="weight")
    else:
        if isinstance(weights, dict):
            weights = pd.Series(weights, dtype=float)
        if isinstance(weights, pd.Series):
            row = weights.astype(float)
        else:
            arr = np.asarray(weights, dtype=float).ravel()
            if len(arr) != len(cols):
                raise ValueError(f"{len(arr)} weights for {len(cols)} assets; name them to weight a subset")
            row = pd.Series(arr, index=cols)
        start = [index[0] - pd.Timedelta(days=1)] if index is not None else [0]
        df = pd.DataFrame([row.values], index=start, columns=row.index, dtype=float)
    unknown = [a for a in df.columns if a not in cols]
    if unknown:
        raise AlignmentError(f"weights name assets absent from returns: {unknown}")
    df = df[[a for a in cols if a in df.columns]]
    if df.isna().any().any():
        raise ParisError("weights contain NaN; every asset needs a weight on every weight date")
    sums = df.sum(axis=1)
    if normalize:
        if (sums.abs() < tol).any():
            raise ParisError("a weight row sums to zero and cannot be normalised")
        df = df.div(sums, axis=0)
    elif ((sums - 1.0).abs() > tol).any():
        bad = sums[(sums - 1.0).abs() > tol]
        raise ParisError(f"weights must sum to 1 within {tol:g}; got {bad.round(6).to_dict()} "
                         "(pass normalize=True to rescale)")
    return df


# --------------------------------------------------------------------------- frequency
def infer_periods(index: pd.Index) -> int:
    """Infer periods-per-year from median spacing of a DatetimeIndex (252/52/12/4/1)."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise FrequencyError("need a DatetimeIndex with >= 2 observations to infer frequency; "
                             "pass periods_per_year explicitly")
    days = float(np.median(np.diff(index.values).astype("timedelta64[s]").astype(float)) / 86400.0)
    for lo, hi, ppy in _FREQ_BANDS:
        if lo <= days <= hi:
            return ppy
    raise FrequencyError(f"cannot infer frequency from median spacing of {days:.1f} days; "
                         "pass periods_per_year explicitly")


def resolve_periods(index: pd.Index, periods_per_year: int | None) -> int:
    if periods_per_year is None:
        return infer_periods(index)
    if periods_per_year <= 0:
        raise FrequencyError("periods_per_year must be positive")
    return int(periods_per_year)


# --------------------------------------------------------------------------- alignment
def _check_gaps(df: pd.DataFrame, label: str) -> None:
    bad = df.index[df.isna().any(axis=1)]
    if len(bad):
        cols = [c for c in df.columns if df[c].isna().any()]
        shown = ", ".join(str(d.date() if hasattr(d, "date") else d) for d in bad[:5])
        raise GapError(f"{label} has {len(bad)} interior gap(s) in {cols} (e.g. {shown}); "
                       "gaps must be resolved upstream, PARIS never fills or drops them")


def _side_series(x: Any, index: pd.Index, name: str) -> pd.Series:
    """Coerce a benchmark / rf input. A bare array or list carries no index of its own, so it is
    aligned to ``returns`` by position and must have the same length; a Series keeps its index."""
    s = to_series(x, name)
    if isinstance(x, (pd.Series, pd.DataFrame)) or hasattr(x, "to_pandas"):
        return s
    if len(s) != len(index):
        raise AlignmentError(f"{name} array length {len(s)} != returns length {len(index)}")
    return pd.Series(s.values, index=index, name=s.name)


def align(
    returns: Any, benchmark: Any = None, rf: Any = None
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Trim all inputs to their common window [max first-valid, min last-valid].

    Returns (returns_df, benchmark_series|None, rf_series|None). Scalar / None rf passes through
    as None. A bare array / list benchmark or rf is aligned to ``returns`` by position. Interior
    NaN or missing dates inside the common window raise :class:`GapError`.
    """
    df = to_frame(returns)
    bench = _side_series(benchmark, df.index, "benchmark") if benchmark is not None else None
    rfs = _side_series(rf, df.index, "rf") if (rf is not None and not np.isscalar(rf)) else None
    parts = [df] + [s.to_frame() for s in (bench, rfs) if s is not None]
    starts = [col.first_valid_index() for p in parts for _, col in p.items()]
    ends = [col.last_valid_index() for p in parts for _, col in p.items()]
    if any(s is None for s in starts) or any(e is None for e in ends):
        raise AlignmentError("an input series contains no valid observations")
    start, end = max(starts), min(ends)
    if start > end:
        raise AlignmentError("inputs share no common window")
    df = df.loc[start:end]
    _check_gaps(df, "returns")
    if bench is not None:
        bench = bench.reindex(df.index)
        _check_gaps(bench.to_frame(), "benchmark")
    if rfs is not None:
        rfs = rfs.reindex(df.index)
        _check_gaps(rfs.to_frame(), "rf")
    return df, bench, rfs


# --------------------------------------------------------------------------- risk-free
def rf_per_period(rf: Any, index: pd.Index, ppy: int, compounding: bool = True) -> pd.Series:
    """Per-period risk-free series aligned to *index*.

    Scalar rf is an ANNUAL rate, de-annualised geometrically ((1+rf)^(1/ppy)-1) or
    arithmetically (rf/ppy) when ``compounding=False``. A Series/array is taken as already periodic.
    """
    if rf is None:
        rf = 0.0
    if np.isscalar(rf):
        per = (1.0 + float(rf)) ** (1.0 / ppy) - 1.0 if compounding else float(rf) / ppy
        return pd.Series(per, index=index, name="rf")
    if isinstance(rf, (pd.Series, pd.DataFrame)) or hasattr(rf, "to_pandas"):
        s = to_series(rf, "rf").reindex(index)
        _check_gaps(s.to_frame(), "rf")
        return s.rename("rf")
    arr = np.asarray(rf, dtype=float).ravel()
    if len(arr) != len(index):
        raise AlignmentError(f"rf array length {len(arr)} != returns length {len(index)}")
    return pd.Series(arr, index=index, name="rf")


# --------------------------------------------------------------------------- annualisation
def annualize_return(r: pd.Series, ppy: int, geometric: bool = True) -> float:
    """Annualised return of a periodic return series."""
    n = len(r)
    if n == 0:
        return float("nan")
    if geometric:
        return float(np.prod(1.0 + r.values) ** (ppy / n) - 1.0)
    return float(r.mean() * ppy)


def annualize_vol(sd: float, ppy: int) -> float:
    return float(sd * np.sqrt(ppy))


def rf_annual(rf: Any, index: pd.Index, periods_per_year: int | None) -> float:
    """Annual risk-free rate as a scalar, for ratios whose numerator is an annualised return.

    A scalar is returned unchanged (it is already annual). A per-period Series/array is aligned to
    *index* and annualised geometrically, ``prod(1+rf)^(ppy/n)-1`` — the same rule used for the
    fund return — which requires ``ppy`` (inferred from the index when not given).
    """
    if rf is None:
        return 0.0
    if np.isscalar(rf):
        return float(rf)
    ppy = resolve_periods(index, periods_per_year)
    return annualize_return(rf_per_period(rf, index, ppy), ppy)


# --------------------------------------------------------------------------- prepare
@dataclass
class Prepared:
    """Aligned, validated inputs ready for metric computation."""

    returns: pd.DataFrame
    benchmark: pd.Series | None
    rf: pd.Series
    ppy: int
    multi: bool  # caller passed several series -> return Series of results, else scalar

    @property
    def excess(self) -> pd.DataFrame:
        return self.returns.sub(self.rf, axis=0)

    @property
    def active(self) -> pd.DataFrame | None:
        return None if self.benchmark is None else self.returns.sub(self.benchmark, axis=0)

    @property
    def bench_excess(self) -> pd.Series | None:
        return None if self.benchmark is None else self.benchmark - self.rf


def prepare(
    returns: Any,
    benchmark: Any = None,
    rf: Any = 0.0,
    periods_per_year: int | None = None,
    compounding: bool = True,
) -> Prepared:
    multi = is_multi(returns)
    df, bench, rfs = align(returns, benchmark, rf)
    ppy = resolve_periods(df.index, periods_per_year)
    rf_series = rfs if rfs is not None else rf_per_period(rf, df.index, ppy, compounding)
    return Prepared(df, bench, rf_series, ppy, multi)


def apply_cols(fn: Callable[..., Any], returns: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply a Series->scalar function per column; scalar for Series input, Series for DataFrame.
    Undefined ratios (0/0) yield NaN/inf silently rather than numpy warnings."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if isinstance(returns, pd.DataFrame):
            return pd.Series({col: fn(returns[col], *args, **kwargs) for col in returns.columns},
                             dtype=float if returns.shape[1] else object)
        return fn(returns, *args, **kwargs)


def result(p: Prepared, fn: Callable[[pd.Series], Any], frame: pd.DataFrame | None = None) -> Any:
    """Run *fn* over each column of ``frame`` (default ``p.returns``); unwrap to scalar if not multi."""
    frame = p.returns if frame is None else frame
    out = apply_cols(fn, frame)
    return out if p.multi else out.iloc[0]


# --------------------------------------------------------------------------- numerical helpers
# Shared by risk.py and ratios.py so that each topic module depends on this file only.
def _downside(s: pd.Series, mar: pd.Series, method: str, power: int = 2, annualize_ppy: int = 1) -> float:
    d = np.minimum(s.values - mar.values, 0.0)
    if method == "full":
        n = len(d)
    elif method == "subset":
        n = int((s.values < mar.values).sum())
    else:
        raise ValueError("method must be 'full' or 'subset'")
    if n == 0:
        return float("nan")
    return float((np.sum(np.abs(d) ** power) / n) ** (1.0 / power) * math.sqrt(annualize_ppy))



def _central(x: np.ndarray, k: int) -> float:
    return float(np.mean((x - x.mean()) ** k))


def _std_moment(x: np.ndarray, k: int) -> float:
    """Standardised population moment m_k / m_2^(k/2). A constant series has no dispersion, so its
    skewness and kurtosis are 0/0: NaN (never an exception)."""
    if len(x) < 2 or np.ptp(x) == 0.0:
        return float("nan")
    return _central(x, k) / _central(x, 2) ** (k / 2)


def is_singular(x: np.ndarray) -> bool:
    """True for a series without dispersion: fewer than two observations or zero range. Tested on
    the range, not the variance, so a constant series is never a quotient of rounding residues."""
    return len(x) < 2 or np.ptp(x) == 0.0


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS slope and intercept of y on x; ``(NaN, NaN)`` for a singular regressor (no
    dispersion in x: the slope is undefined)."""
    if is_singular(x):
        return float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    b = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    return b, float(ym - b * xm)


def pearson(y: np.ndarray, x: np.ndarray) -> float:
    """Pearson correlation; NaN when either series has no dispersion."""
    if is_singular(x) or is_singular(y):
        return float("nan")
    return float(np.corrcoef(y, x)[0, 1])


def pop_sd(x: pd.Series) -> float:
    """Population sd, exactly 0 for a constant series (pandas returns a rounding residue)."""
    return 0.0 if is_singular(x.values) else float(x.std(ddof=0))


def safe_div(num: float, den: float) -> float:
    """Division with IEEE / R semantics: x/0 -> +-inf, 0/0 -> NaN, never ``ZeroDivisionError``
    (manual section 1.7: one degenerate column must not abort a table of many funds)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(num) / np.float64(den))



_SQRT2PI = math.sqrt(2 * math.pi)


def _pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / _SQRT2PI


def _cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _ppf(q: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation refined by one Newton step)."""
    if not 0 < q < 1:
        raise ValueError("confidence must be in (0, 1)")
    if q > 0.5:  # symmetry keeps full precision in the upper tail
        return -_ppf(1 - q)
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    if q < 0.02425:
        t = math.sqrt(-2 * math.log(q))
        x = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / \
            ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    else:
        t = q - 0.5
        r = t * t
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = _cdf(x) - q  # one Newton refinement -> ~1e-15
    return x - e * _SQRT2PI * math.exp(x * x / 2)


def _cf_z(z: float, s: float, k: float) -> float:
    """Cornish-Fisher adjusted quantile (``s`` skewness, ``k`` excess kurtosis)."""
    return z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36


def _var_series(s: pd.Series, confidence: float, method: str, interpolation: str, ddof: int) -> float:
    x = s.values
    if method == "historical":
        return float(np.quantile(x, 1 - confidence, method=interpolation))
    mu, sd = x.mean(), x.std(ddof=ddof)
    z = _ppf(1 - confidence)
    if method == "gaussian":
        return float(mu + z * sd)
    if method == "modified":
        sk, ek = _std_moment(x, 3), _std_moment(x, 4) - 3  # NaN for a constant series
        return float(mu + _cf_z(z, sk, ek) * sd)
    raise ValueError("method must be 'historical', 'gaussian' or 'modified'")



def _es_series(s: pd.Series, confidence: float, method: str, interpolation: str, ddof: int,
               operational: bool = True) -> float:
    x = s.values
    alpha = 1 - confidence
    if method in ("historical", "gaussian_tail"):
        q = (np.quantile(x, alpha, method=interpolation) if method == "historical"
             else x.mean() + _ppf(alpha) * x.std(ddof=ddof))
        tail = x[x < q]
        return float(tail.mean()) if len(tail) else float(q)  # no exceedances -> ES = VaR
    mu, sd = x.mean(), x.std(ddof=ddof)
    z = _ppf(alpha)
    if method == "gaussian":
        return float(mu - sd * _pdf(z) / alpha)
    if method == "modified":
        sk, ek = _std_moment(x, 3), _std_moment(x, 4) - 3
        if math.isnan(sk):  # constant series: the Cornish-Fisher expansion is undefined
            return float("nan")
        h = _cf_z(z, sk, ek)
        e = _pdf(h) * (1 + h**3 * sk / 6 + (h**6 - 9 * h**4 + 9 * h**2 + 3) * sk**2 / 72
                       + (h**4 - 2 * h**2 - 1) * ek / 24)
        mes, mvar = mu - sd * e / alpha, mu + h * sd
        return float(min(mes, mvar)) if operational else float(mes)
    raise ValueError("method must be 'historical', 'gaussian', 'modified' or 'gaussian_tail'")


