"""Too few or degenerate observations give NaN (or +-inf where the reference convention is a
signed division by zero) - never an exception - on slices of the sample data and trivial
constant series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import paris

PPY = {"periods_per_year": 12}


@pytest.fixture(scope="module")
def const(fcntx):
    return pd.Series(0.05, index=fcntx.index[:24])


@pytest.fixture(scope="module")
def zeros(fcntx):
    return pd.Series(0.0, index=fcntx.index[:24])


@pytest.fixture(scope="module")
def rising(fcntx):
    return pd.Series(np.linspace(0.01, 0.03, 24), index=fcntx.index[:24])


def test_sample_moments_need_enough_observations(fcntx):
    two, three = fcntx.iloc[:2], fcntx.iloc[:3]
    for method in ("sample", "fisher"):
        assert np.isnan(paris.skewness(two, method=method)), method
    for method in ("sample", "sample_excess", "fisher"):
        assert np.isnan(paris.kurtosis(three, method=method)), method
    assert np.isfinite(paris.kurtosis(three))  # population form is defined for n = 3
    assert np.isfinite(paris.skewness(three))


def test_moments_of_a_constant_series_are_nan(const):
    for method in ("moment", "fisher", "sample"):
        assert np.isnan(paris.skewness(const, method=method)), method
    for method in ("excess", "moment", "sample", "sample_excess", "fisher"):
        assert np.isnan(paris.kurtosis(const, method=method)), method
    assert paris.skewness(pd.DataFrame({"A": const, "B": const})).isna().all()
    assert np.isnan(paris.adjusted_sharpe(const)) and np.isnan(paris.skewness_kurtosis_ratio(const))
    assert np.isnan(paris.probabilistic_sharpe(const)) and np.isnan(
        paris.probabilistic_sharpe(const, method="moment")
    )


def test_var_of_a_constant_series(const):
    assert np.isclose(paris.var(const), 0.05) and np.isclose(
        paris.var(const, method="gaussian"), 0.05
    )
    assert np.isclose(paris.cvar(const), 0.05) and np.isclose(
        paris.cvar(const, method="gaussian"), 0.05
    )
    # Cornish-Fisher forms need skewness and kurtosis, undefined here
    assert np.isnan(paris.var(const, method="modified"))
    assert np.isnan(paris.cvar(const, method="modified"))
    assert np.isnan(paris.cvar(const, method="modified", operational=False))


def test_one_sided_statistics_with_an_empty_side(rising):
    losses = -rising
    assert np.isnan(paris.avg_loss(rising)) and np.isnan(paris.avg_win(losses))
    assert np.isnan(paris.loss_deviation(rising)) and np.isnan(paris.gain_deviation(losses))
    assert np.isnan(paris.gain_deviation(pd.Series([0.1, -0.2, -0.3])))  # one gain: sd undefined
    assert np.isnan(paris.downside_deviation(rising, method="subset"))
    assert np.isnan(paris.upside_risk(losses, method="subset"))
    assert paris.win_rate(rising) == 1.0 and paris.win_rate(losses) == 0.0
    assert paris.consecutive_losses(rising) == 0 and paris.consecutive_wins(rising) == len(rising)


def test_no_drawdown_series(rising):
    assert paris.drawdowns(rising).eq(0).all()
    assert paris.max_drawdown(rising) == 0 and paris.longest_drawdown(rising) == 0
    assert np.isnan(paris.avg_drawdown(rising)) and np.isnan(paris.avg_recovery(rising))
    assert np.isnan(paris.burke_ratio(rising, **PPY))
    assert len(paris.drawdown_table(rising)) == 0
    assert paris.pain_index(rising) == 0 and paris.ulcer_index(rising) == 0


def test_zero_over_zero_ratios_are_nan(zeros, const):
    for fn in (paris.calmar_ratio, paris.pain_ratio, paris.martin_ratio, paris.recovery_factor):
        assert np.isnan(fn(zeros)), fn.__name__
    assert (
        np.isnan(paris.sharpe(zeros))
        and np.isnan(paris.sortino(zeros))
        and np.isnan(paris.omega(zeros))
    )
    # a positive return with no drawdown or downside: division by zero is +inf
    assert np.isposinf(paris.sortino(const))
    assert np.isposinf(paris.omega(const))
    # a return below rf with a zero denominator is -inf
    assert np.isneginf(paris.pain_ratio(const, rf=0.99)) and np.isneginf(
        paris.martin_ratio(const, rf=0.99)
    )


def test_constant_benchmark_makes_regression_undefined(fcntx):
    cb = pd.Series(0.01, index=fcntx.index)
    for fn in (
        paris.beta,
        paris.alpha,
        paris.jensen_alpha,
        paris.bull_beta,
        paris.bear_beta,
        paris.treynor_ratio,
        paris.timing_ratio,
        paris.appraisal_ratio,
        paris.systematic_risk,
        paris.specific_risk,
        paris.total_risk,
        paris.net_selectivity,
    ):
        assert np.isnan(fn(fcntx, cb, **PPY)), fn.__name__
    assert np.isnan(paris.correlation(fcntx, cb)) and np.isnan(paris.r_squared(fcntx, cb))
    assert np.isinf(paris.fama_beta(fcntx, cb, **PPY))  # sd(fund) / 0
    row = paris.regression_stats(fcntx, cb, **PPY).iloc[0]
    assert row["n"] == len(fcntx) and row.drop("n").isna().all()
    assert paris.tracking_error(fcntx, fcntx) == 0  # zero, not NaN
    assert np.isnan(paris.information_ratio(fcntx, fcntx))  # 0 / 0


def test_treynor_sign_conventions(fcntx):
    b = fcntx.iloc[:24]
    zeros = pd.Series(0.0, index=b.index)
    assert np.isnan(paris.treynor_ratio(zeros, b))  # beta 0, excess 0
    half = pd.Series(0.5, index=b.index)  # exactly representable: beta exactly 0
    assert np.isposinf(paris.treynor_ratio(half, b))
    assert np.isposinf(paris.treynor_ratio(half, b, modified=True))


def test_period_returns_with_insufficient_history(fcntx):
    short = paris.period_returns(fcntx.iloc[-30:])
    assert np.isnan(short.loc["3Y", "FCNTX"]) and np.isnan(short.loc["10Y", "FCNTX"])
    assert np.isfinite(short.loc["1Y", "FCNTX"]) and np.isfinite(short.loc["ITD", "FCNTX"])
    six = paris.period_returns(fcntx.iloc[-6:])
    assert np.isclose(
        six.loc["ITD", "FCNTX"], (1 + fcntx.iloc[-6:]).prod() - 1
    )  # < 1 year: not annualised


def test_rolling_shorter_than_window_is_empty_or_nan(fcntx):
    out = paris.rolling(fcntx.iloc[:36], paris.sharpe, 36)
    assert len(out) == 1
    out = paris.rolling(fcntx.iloc[:40], paris.sharpe, 36, trim=False)
    assert len(out) == 40 and out.iloc[:35].isna().all() and out.iloc[35:].notna().all()


def test_two_observations(fcntx):
    two = fcntx.iloc[:2]
    assert np.isfinite(paris.volatility(two)) and np.isfinite(paris.total_return(two))
    assert not np.isnan(paris.volatility(two, ddof=1))
    assert np.isnan(paris.volatility(fcntx.iloc[:1], **PPY))
