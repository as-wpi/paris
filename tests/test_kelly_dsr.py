"""Kelly leverage variants and interval, deflated Sharpe ratio and minimum track record length
against hand computations and the identities that tie them to the existing functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import paris
from paris._core import _ppf

EULER_GAMMA = 0.5772156649015329


# ------------------------------------------------------------------ Kelly
def test_kelly_scaling_and_denominator(funds, rf_series):
    k = paris.kelly_ratio(funds, rf=rf_series)
    np.testing.assert_allclose(paris.kelly_ratio(funds, rf=rf_series, half=True), k / 2)
    np.testing.assert_allclose(paris.kelly_ratio(funds, rf=rf_series, fraction=0.5), k / 2)
    np.testing.assert_allclose(paris.kelly_ratio(funds, rf=rf_series, half=True, fraction=0.5), k / 4)
    ex = funds.sub(rf_series, axis=0)
    np.testing.assert_allclose(k, ex.mean() / funds.var(ddof=1))
    np.testing.assert_allclose(paris.kelly_ratio(funds, rf=rf_series, excess_var=True), ex.mean() / ex.var(ddof=1))
    assert paris.kelly_ratio(funds).equals(paris.kelly_ratio(funds, excess_var=True))  # rf = 0: same


def test_kelly_is_frequency_invariant(daily):
    daily_k = paris.kelly_ratio(daily["SPY"])
    monthly = paris.aggregate(daily["SPY"], "ME")
    monthly_k = paris.kelly_ratio(monthly)
    assert abs(daily_k - monthly_k) / abs(daily_k) < 0.5  # same order: both estimate mu / sigma^2


def test_kelly_interval_matches_the_delta_method(funds, fcntx, rf_series):
    t = paris.kelly_interval(funds, rf=rf_series)
    assert list(t.index) == ["lower", "kelly", "upper"] and list(t.columns) == list(funds.columns)
    np.testing.assert_allclose(t.loc["kelly"], paris.kelly_ratio(funds, rf=rf_series))
    n = len(fcntx)
    m, v = (fcntx - rf_series).mean(), fcntx.var(ddof=1)
    se = np.sqrt(1 / (n * v) + 2 * m**2 / (v**2 * (n - 1)))
    z = _ppf(0.975)
    assert np.isclose(z, 1.959963984540054, atol=1e-9)
    one = paris.kelly_interval(fcntx, rf=rf_series)
    np.testing.assert_allclose(one.to_numpy(), [m / v - z * se, m / v, m / v + z * se])
    half = paris.kelly_interval(fcntx, rf=rf_series, half=True)
    np.testing.assert_allclose(half.to_numpy(), one.to_numpy() / 2)
    narrow = paris.kelly_interval(fcntx, rf=rf_series, confidence=0.5)
    assert narrow["upper"] - narrow["lower"] < one["upper"] - one["lower"]
    # sixteen years of monthly data leave a wide band: the sign is the only robust statement
    assert (t.loc["upper"] - t.loc["lower"]).min() > 3.0


# ------------------------------------------------------------------ deflated Sharpe
def test_deflated_sharpe_reduces_to_psr_without_selection(funds, rf_series):
    np.testing.assert_allclose(paris.deflated_sharpe(funds, rf=rf_series),
                               paris.probabilistic_sharpe(funds, rf=rf_series))
    np.testing.assert_allclose(paris.deflated_sharpe(funds, trials=10, sharpe_variance=0.0),
                               paris.probabilistic_sharpe(funds))


def test_deflated_sharpe_uses_the_expected_maximum(funds, fcntx):
    n_trials, var = 10, 0.01
    sr0 = np.sqrt(var) * ((1 - EULER_GAMMA) * _ppf(1 - 1 / n_trials) + EULER_GAMMA * _ppf(1 - 1 / (n_trials * np.e)))
    assert 0.1 < sr0 < 0.2  # about 1.5 sd above zero for ten trials
    np.testing.assert_allclose(paris.deflated_sharpe(funds, trials=n_trials, sharpe_variance=var),
                               paris.probabilistic_sharpe(funds, benchmark_sharpe=sr0))
    assert (paris.deflated_sharpe(funds, trials=n_trials, sharpe_variance=var) < paris.probabilistic_sharpe(funds)).all()
    srs = [0.05, 0.10, 0.20, 0.15, 0.30]
    np.testing.assert_allclose(
        paris.deflated_sharpe(fcntx, trials=srs),
        paris.deflated_sharpe(fcntx, trials=len(srs), sharpe_variance=np.var(srs, ddof=1)),
    )


# ------------------------------------------------------------------ minimum track record
def test_min_track_record_matches_the_closed_form_and_psr(funds, fcntx, rf_series):
    x = fcntx - rf_series
    sr, g3, g4, n = x.mean() / x.std(ddof=1), x.skew(), x.kurt() + 3, len(x)
    z = _ppf(0.95)
    exp = 1 + (1 - g3 * sr + (g4 - 1) / 4 * sr**2) * (z / sr) ** 2
    assert np.isclose(paris.min_track_record(fcntx, rf=rf_series), exp)
    assert np.isclose(paris.min_track_record(fcntx, rf=rf_series, years=True), exp / 12)
    # consistency with the probabilistic Sharpe ratio: enough history <=> PSR at the confidence
    mtrl = paris.min_track_record(funds, rf=rf_series)
    psr = paris.probabilistic_sharpe(funds, rf=rf_series)
    assert ((mtrl <= len(funds)) == (psr >= 0.95)).all()
    assert (paris.min_track_record(funds, rf=rf_series, confidence=0.99) > mtrl).all()
    assert np.isposinf(paris.min_track_record(fcntx, benchmark_sharpe=10.0))
    assert np.isnan(paris.min_track_record(pd.Series(0.01, index=fcntx.index[:24])))


@pytest.mark.parametrize(
    "call",
    [
        lambda r: paris.deflated_sharpe(r, trials=0),
        lambda r: paris.deflated_sharpe(r, trials=5),  # count > 1 without sharpe_variance
        lambda r: paris.deflated_sharpe(r, trials=[0.1]),
        lambda r: paris.deflated_sharpe(r, trials=[0.1, 0.2], sharpe_variance=0.01),
        lambda r: paris.deflated_sharpe(r, trials=5, sharpe_variance=-1.0),
        lambda r: paris.deflated_sharpe(r, trials=5, sharpe_variance=0.01, method="bogus"),
        lambda r: paris.min_track_record(r, confidence=1.0),
        lambda r: paris.min_track_record(r, method="bogus"),
        lambda r: paris.kelly_interval(r, confidence=0.0),
    ],
)
def test_kelly_dsr_switches_raise_value_error(call, fcntx):
    with pytest.raises(ValueError):
        call(fcntx)


def test_stats_table_labels_the_kelly_row_as_leverage(funds):
    t = paris.stats(funds)
    assert "Kelly leverage (half)" in t.index and "Kelly (half)" not in t.index
    np.testing.assert_allclose(t.loc["Kelly leverage (half)"].astype(float), paris.kelly_ratio(funds, half=True))
