"""The sample datasets shipped in ``paris.data``: shape, calendar, rectangularity and the
consistency of the daily index levels with the monthly returns. No network; the CSVs are frozen."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import paris
from paris import data
from paris._core import prepare
from tests.conftest import BENCHMARKS, FUNDS


@pytest.fixture(scope="module")
def managers():
    return data.load_managers()


@pytest.fixture(scope="module")
def prices():
    return data.load_prices()


def test_managers_shape_and_columns(managers):
    assert list(managers.columns) == FUNDS + BENCHMARKS + ["TBILL3M"]
    assert managers.shape == (192, 10)
    assert managers.index.name == "date"
    assert isinstance(managers.index, pd.DatetimeIndex)
    assert managers.dtypes.eq(float).all()


def test_managers_calendar_is_every_month_end(managers):
    assert managers.index.equals(pd.date_range("2010-01-31", "2025-12-31", freq="ME"))


def test_managers_has_no_missing_values(managers):
    assert not managers.isna().any().any()


def test_managers_magnitudes(managers):
    r = managers[FUNDS + BENCHMARKS]
    assert (r.abs() < 0.5).all().all()
    assert (r != 0).mean().min() > 0.99  # no stale (zero-return) months beyond rounding
    rf = managers["TBILL3M"]
    assert ((rf >= 0) & (rf < 0.006)).all()  # per month: never above ~7 % a year


def test_managers_is_rectangular_under_prepare(managers):
    p = prepare(managers[FUNDS], benchmark=managers["SPY"], rf=managers["TBILL3M"])
    assert len(p.returns) == 192  # nothing trimmed: full common window


def test_prices_shape_and_calendar(prices):
    assert list(prices.columns) == ["SPY", "FCNTX"]
    assert prices.index.name == "date"
    assert prices.index.is_monotonic_increasing and not prices.index.has_duplicates
    assert prices.index[0] == pd.Timestamp("2021-01-04")
    assert prices.index[-1] == pd.Timestamp("2025-12-31")
    assert 1200 <= len(prices) <= 1300  # ~252 trading days x 5 years
    assert not prices.isna().any().any()
    assert (prices > 0).all().all()
    spacing = np.diff(prices.index.values).astype("timedelta64[D]").astype(int)
    assert spacing.max() <= 5  # no multi-week holes


def test_prices_monthly_returns_match_managers(prices, managers):
    me = prices.resample("ME").last().pct_change().iloc[1:]
    m = managers.loc[me.index, ["SPY", "FCNTX"]]
    np.testing.assert_allclose(me.to_numpy(), m.to_numpy(), atol=1e-9)


def test_describe_matches_loaders(managers, prices):
    d = data.describe()
    assert list(d.columns) == ["dataset", "column", "name", "role", "frequency", "first", "last"]
    assert d.loc[d.dataset == "managers", "column"].tolist() == list(managers.columns)
    assert d.loc[d.dataset == "prices", "column"].tolist() == list(prices.columns)
    assert (d.loc[d.dataset == "managers", "first"] == "2010-01-31").all()
    assert (d.loc[d.dataset == "prices", "last"] == "2025-12-31").all()
    assert set(d.role) == {"fund", "benchmark", "risk-free"}


def test_loaders_return_fresh_copies(managers):
    managers.iloc[0, 0] = 123.0
    assert data.load_managers().iloc[0, 0] != 123.0


def test_stats_runs_on_the_sample_data(managers):
    t = paris.stats(managers[FUNDS], benchmark=managers["SPY"], rf=managers["TBILL3M"])
    assert list(t.columns) == FUNDS + ["SPY"]
    assert not t.loc[["CAGR", "Volatility (ann.)", "Sharpe", "Beta"]].isna().any().any()
