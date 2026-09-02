"""0.6.0 extensions: rolling Ulcer, drawdown distribution, windowed and calendar-basis drawdown
ratios, Kelly leverage variants and interval, deflated Sharpe and minimum track record — checked
against hand computations and the identities that tie them to the existing functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import paris

PPY = {"periods_per_year": 12}


# ------------------------------------------------------------------ Ulcer & distributions
def test_ulcer_pct_is_a_scale(funds):
    np.testing.assert_allclose(paris.ulcer_index(funds, pct=True), 100 * paris.ulcer_index(funds))


def test_rolling_ulcer_is_the_rms_of_full_history_drawdowns(fcntx, funds):
    dd = paris.drawdowns(fcntx)
    manual = np.sqrt((dd**2).rolling(36).mean()).dropna()
    pd.testing.assert_series_equal(paris.rolling_ulcer(fcntx, 36), manual, check_names=False)
    full = paris.rolling_ulcer(fcntx, len(fcntx))
    assert len(full) == 1 and np.isclose(full.iloc[0], paris.ulcer_index(fcntx))
    untrimmed = paris.rolling_ulcer(fcntx, 36, trim=False)
    assert len(untrimmed) == len(fcntx) and untrimmed.iloc[:35].isna().all()
    frame = paris.rolling_ulcer(funds, 36)
    assert list(frame.columns) == list(funds.columns)
    pd.testing.assert_series_equal(frame["FCNTX"], paris.rolling_ulcer(fcntx, 36), check_names=False)
    np.testing.assert_allclose(paris.rolling_ulcer(fcntx, 36, pct=True), 100 * paris.rolling_ulcer(fcntx, 36))
    # ddof=1 divides by window-1
    np.testing.assert_allclose(
        paris.rolling_ulcer(fcntx, 36, ddof=1), paris.rolling_ulcer(fcntx, 36) * np.sqrt(36 / 35)
    )


def test_rolling_ulcer_is_exactly_zero_at_the_peak(funds):
    # the sliding-window sum leaves ~1e-19 residues once a drawdown leaves the window
    for kw in ({}, {"geometric": False}, {"ddof": 1, "geometric": False}):
        out = paris.rolling_ulcer(funds, 12, **kw)
        assert not out.isna().any().any(), kw
        assert (out >= 0).all().all()
    rising = pd.Series(np.linspace(0.01, 0.03, 24), index=funds.index[:24])
    assert (paris.rolling_ulcer(rising, 6) == 0).all()


def test_rolling_ulcer_agrees_with_generic_rolling_outside_drawdowns(fcntx):
    # generic rolling re-bases the peak inside each window, so it can only be smaller
    generic = paris.rolling(fcntx, paris.ulcer_index, 36)
    dedicated = paris.rolling_ulcer(fcntx, 36)
    assert (dedicated + 1e-15 >= generic).all()
    assert np.isclose(dedicated.max(), generic.max())  # the deepest window starts at a peak


def test_drawdown_distribution_ties_to_the_scalar_functions(funds, fcntx):
    t = paris.drawdown_distribution(funds)
    assert list(t.index) == ["share |x| < 0.01", "mean", "q50", "q75", "q90", "q95", "q99", "max"]
    assert list(t.columns) == list(funds.columns)
    np.testing.assert_allclose(t.loc["max"], paris.max_drawdown(funds))
    np.testing.assert_allclose(t.loc["mean"], -paris.drawdowns(funds).abs().mean())
    assert ((t.loc["share |x| < 0.01"] >= 0) & (t.loc["share |x| < 0.01"] <= 1)).all()
    q = t.loc[["q50", "q75", "q90", "q95", "q99", "max"]]
    assert (q.diff().dropna() <= 1e-15).all().all()  # more negative down the rows
    u = paris.drawdown_distribution(fcntx, stat="ulcer", window=36)
    assert np.isclose(u.loc["max", "FCNTX"], paris.rolling_ulcer(fcntx, 36).max())
    assert (u.loc[["mean", "q50", "max"]] >= 0).all().all()
    d = paris.drawdown_distribution(fcntx)  # a Series gives one column
    assert list(d.columns) == ["FCNTX"]


# ------------------------------------------------------------------ windowed & calendar ratios
def test_window_restricts_to_the_trailing_observations(funds, fcntx):
    np.testing.assert_allclose(paris.calmar_ratio(funds, window=36), paris.calmar_ratio(funds.iloc[-36:]))
    np.testing.assert_allclose(paris.sterling_ratio(funds, window=36), paris.sterling_ratio(funds.iloc[-36:]))
    assert np.isclose(paris.calmar_ratio(fcntx, window=len(fcntx)), paris.calmar_ratio(fcntx))


def test_calendar_basis_reconciles_with_cagr_and_periods(funds, fcntx):
    ui = paris.ulcer_index(funds)
    cal = paris.cagr(funds, method="calendar", start="2009-12-31")
    np.testing.assert_allclose(paris.martin_ratio(funds, method="calendar", start="2009-12-31"), cal / ui)
    # 2009-12-31 -> 2025-12-31 is 5844 days = exactly 16 years of 365.25 days = 192 months
    np.testing.assert_allclose(
        paris.martin_ratio(funds, method="calendar", start="2009-12-31"), paris.martin_ratio(funds), rtol=1e-12
    )
    np.testing.assert_allclose(
        paris.calmar_ratio(funds, method="calendar", start="2009-12-31"), paris.calmar_ratio(funds), rtol=1e-12
    )
    # without start the window is one month shorter, so the calendar CAGR is a little higher
    assert (paris.martin_ratio(funds, method="calendar") > paris.martin_ratio(funds)).all()
    np.testing.assert_allclose(
        paris.pain_ratio(fcntx, method="calendar"), paris.cagr(fcntx, method="calendar") / paris.pain_index(fcntx)
    )
    np.testing.assert_allclose(
        paris.sterling_ratio(fcntx, method="calendar"),
        paris.cagr(fcntx, method="calendar") / (abs(paris.max_drawdown(fcntx)) + 0.10),
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda r: paris.calmar_ratio(r, window=1),
        lambda r: paris.calmar_ratio(r, window=len(r) + 1),
        lambda r: paris.sterling_ratio(r, window=2.5),
        lambda r: paris.martin_ratio(r, method="years"),
        lambda r: paris.rolling_ulcer(r, 1),
        lambda r: paris.rolling_ulcer(r, len(r) + 1),
        lambda r: paris.drawdown_distribution(r, stat="pain"),
        lambda r: paris.drawdown_distribution(r, quantiles=(0.5, 1.0)),
    ],
)
def test_extension_switches_raise_value_error(call, fcntx):
    with pytest.raises(ValueError):
        call(fcntx)
