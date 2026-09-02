"""``Portfolio`` is sugar: every method delegates to the free function with the wrapper's
returns, benchmark, rf, periods_per_year and weights filled in by parameter name."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import paris
from paris import attribution as A
from paris import budgeting as B
from paris import drawdown, ratios, relative, returns, risk, tables
from tests.conftest import W6, W_SLEEVES, W_STYLE

PUBLIC = sorted(
    set().union(*(m.__all__ for m in (returns, risk, drawdown, ratios, relative, tables)))
)
SKIP = {"rolling", "period_contributions"}  # need a function / a contributions frame as argument


@pytest.mark.parametrize("name", [n for n in PUBLIC if n not in SKIP])
def test_every_public_statistic_delegates(name, funds, spx, rf_series):
    pf = paris.Portfolio(funds, spx, rf=rf_series)
    fn = getattr(paris, name)
    params = inspect.signature(fn).parameters
    kwargs = {}
    if "rf" in params:
        kwargs["rf"] = rf_series
    if name == "risk_premium":
        exp = fn(spx, **kwargs)
    elif "benchmark" in params:
        exp = fn(funds, spx, **kwargs)
    else:
        exp = fn(funds, **kwargs)
    got = getattr(pf, name)()
    if isinstance(exp, (pd.Series, pd.DataFrame)):
        assert type(got) is type(exp) and got.equals(exp)
    else:
        np.testing.assert_allclose(got, exp, rtol=0, equal_nan=True)


def test_rolling_and_stats_delegate(funds, spx, rf_series):
    pf = paris.Portfolio(funds, spx, rf=rf_series)
    pd.testing.assert_frame_equal(
        pf.rolling(paris.beta, 36), paris.rolling(funds, paris.beta, 36, spx, rf=rf_series)
    )
    pd.testing.assert_frame_equal(pf.stats(), paris.stats(funds, spx, rf=rf_series))
    pd.testing.assert_frame_equal(
        pf.stats(include_benchmark=False),
        paris.stats(funds, spx, rf=rf_series, include_benchmark=False),
    )


def test_weights_build_the_portfolio_series(funds, spx):
    pf = paris.Portfolio(funds, spx, rf=0.02, weights=W6, rebalance="QE")
    exp = A.portfolio_return(funds, W6, rebalance="QE")
    assert pf.returns.equals(exp) and pf.assets is funds
    assert np.isclose(pf.sharpe(), paris.sharpe(exp, rf=0.02))
    assert np.isclose(pf.beta(), paris.beta(exp, spx, rf=0.02))
    assert list(pf.stats().columns) == ["Portfolio", "SPY"]
    assert pf.contribution().equals(A.contribution(funds, W6, rebalance="QE"))
    assert pf.bop_weights().equals(A.bop_weights(funds, W6, rebalance="QE"))
    assert pf.eop_weights().equals(A.eop_weights(funds, W6, rebalance="QE"))
    assert pf.period_contributions("QE").equals(
        A.period_contributions(A.contribution(funds, W6, rebalance="QE"), "QE")
    )
    assert pf.volatility_contribution(pct=True).equals(
        B.volatility_contribution(funds, W6, pct=True)
    )
    assert pf.var_contribution(0.99, "modified").equals(
        B.var_contribution(funds, W6, 0.99, "modified")
    )
    assert pf.cvar_contribution(method="gaussian").equals(
        B.cvar_contribution(funds, W6, method="gaussian")
    )
    assert pf.marginal_var().equals(B.marginal_var(funds, W6))
    assert pf.marginal_cvar().equals(B.marginal_cvar(funds, W6))


def test_benchmark_weights_build_the_benchmark_series(sleeves, style):
    pf = paris.Portfolio(sleeves, style, weights=W_SLEEVES, benchmark_weights=W_STYLE)
    assert pf.benchmark.name == "Benchmark" and pf.benchmark_assets is style
    assert np.allclose(pf.benchmark.to_numpy(), A.portfolio_return(style, W_STYLE).to_numpy())
    assert list(pf.stats().columns) == ["Portfolio", "Benchmark"]
    assert pf.brinson(linking="none").equals(
        A.brinson(sleeves, W_SLEEVES, style, W_STYLE, linking="none")
    )
    assert pf.active_contribution().equals(
        A.active_contribution(sleeves, W_SLEEVES, style, W_STYLE)
    )


def test_normalize_and_errors(funds, spx):
    pf = paris.Portfolio(funds, weights=[3, 2, 2, 1, 1, 1], normalize=True)
    assert np.allclose(pf.returns.to_numpy(), A.portfolio_return(funds, W6).to_numpy())
    assert np.allclose(pf.contribution().to_numpy(), A.contribution(funds, W6).to_numpy())
    with pytest.raises(ValueError, match="benchmark"):
        paris.Portfolio(funds, weights=W6, benchmark_weights=[1.0])
    plain = paris.Portfolio(funds, spx)
    with pytest.raises(ValueError, match="without weights"):
        plain.contribution()
    with pytest.raises(ValueError, match="without weights"):
        plain.volatility_contribution()
    with pytest.raises(ValueError, match="benchmark_weights"):
        paris.Portfolio(funds, spx, weights=W6).brinson()
    assert "portfolio_return" not in dir(plain) and "sharpe" in dir(plain)
    with pytest.raises(AttributeError):
        plain.__getattr__("portfolio_return")


def test_wrapper_dispatch_is_restricted_to_public_names(fcntx, spx):
    pf = paris.Portfolio(fcntx, spx, rf=0.02)
    for name in (
        "episodes",
        "prepare",
        "result",
        "to_frame",
        "annualize_return",
        "annualize_vol",
        "rf_annual",
    ):
        assert not hasattr(pf, name) and name not in dir(pf), name
    assert np.isclose(pf.risk_premium(), paris.risk_premium(spx, rf=0.02))
    assert set(PUBLIC) <= set(dir(pf))
    assert all(callable(getattr(pf, n)) for n in PUBLIC)


def test_top_level_exports():
    for name in (
        "portfolio_return",
        "contribution",
        "period_contributions",
        "brinson",
        "active_contribution",
        "volatility_contribution",
        "var_contribution",
        "cvar_contribution",
        "marginal_var",
    ):
        assert name in paris.__all__ and callable(getattr(paris, name))
    assert paris.__version__ == "0.6.0"
    assert paris.data.__all__ == ["describe", "load_managers", "load_prices"]
