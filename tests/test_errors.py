"""Bad input raises a ``ParisError`` subclass (or ``ValueError`` for a bad switch value); nothing
is ever filled, dropped or silently re-weighted."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import paris
from paris import AlignmentError, FrequencyError, GapError, ParisError
from tests.conftest import FUNDS, W6


# ------------------------------------------------------------------ gaps (nothing is filled)
def test_interior_nan_raises_gap_error_naming_the_date(funds):
    holed = funds.copy()
    holed.iloc[50, 0] = np.nan
    with pytest.raises(GapError) as e:
        paris.sharpe(holed)
    assert str(funds.index[50].date()) in str(e.value) and "FCNTX" in str(e.value)
    for fn in (
        paris.volatility,
        paris.max_drawdown,
        paris.stats,
        paris.calendar_table,
        paris.period_returns,
    ):
        with pytest.raises(GapError):
            fn(holed)


def test_missing_date_inside_the_window_is_a_gap(funds, fcntx, spx):
    bench = spx.drop(spx.index[100])
    with pytest.raises(GapError):
        paris.beta(fcntx, bench)
    with pytest.raises(GapError):
        paris.rolling(fcntx, paris.beta, 36, bench)
    with pytest.raises(AlignmentError):  # dates differ inside the common window
        paris.active_contribution(funds.drop(funds.index[7]), W6, spx.to_frame(), [1.0])


def test_leading_and_trailing_nan_are_trimmed_not_gaps(fcntx, spx):
    padded = fcntx.copy()
    padded.iloc[:3] = np.nan
    padded.iloc[-2:] = np.nan
    assert np.isclose(paris.sharpe(padded), paris.sharpe(fcntx.iloc[3:-2]))
    assert np.isclose(paris.beta(padded, spx), paris.beta(fcntx.iloc[3:-2], spx))


def test_rf_series_with_a_gap_raises(fcntx, rf_series):
    with pytest.raises(GapError):
        paris.sharpe(fcntx, rf=rf_series.drop(rf_series.index[20]))


# ------------------------------------------------------------------ alignment
def test_unsorted_or_duplicate_index_raises(fcntx):
    with pytest.raises(AlignmentError):
        paris.sharpe(fcntx.iloc[::-1])
    dup = pd.concat([fcntx.iloc[:5], fcntx.iloc[4:6]])
    with pytest.raises(AlignmentError):
        paris.sharpe(dup)


def test_no_common_window_raises(fcntx, spx):
    with pytest.raises(AlignmentError):
        paris.beta(fcntx.iloc[:50], spx.iloc[100:])


def test_array_inputs_must_match_in_length(fcntx, spx):
    with pytest.raises(AlignmentError, match="length"):
        paris.beta(fcntx, spx.to_numpy()[:-1])
    with pytest.raises(AlignmentError, match="length"):
        paris.sharpe(fcntx, rf=np.full(len(fcntx) - 1, 0.001))


def test_two_column_benchmark_is_rejected(funds, m):
    with pytest.raises(AlignmentError, match="single series"):
        paris.beta(funds, m[["SPY", "IWF"]])


def test_non_numeric_input_is_rejected():
    with pytest.raises(ParisError, match="numeric"):
        paris.sharpe(pd.Series(["a", "b", "c"]))


def test_all_nan_series_is_rejected(fcntx):
    with pytest.raises(AlignmentError):
        paris.sharpe(pd.Series(np.nan, index=fcntx.index))


# ------------------------------------------------------------------ frequency
def test_frequency_cannot_be_inferred_without_dates(fcntx):
    with pytest.raises(FrequencyError):
        paris.sharpe(fcntx.to_numpy())  # RangeIndex: no frequency
    assert np.isfinite(paris.sharpe(fcntx.to_numpy(), periods_per_year=12))
    with pytest.raises(FrequencyError):
        paris.sharpe(fcntx.iloc[:1])  # one observation
    with pytest.raises(FrequencyError):
        paris.volatility(fcntx, periods_per_year=0)


def test_calendar_table_needs_monthly_or_finer(fcntx):
    with pytest.raises(FrequencyError):
        paris.calendar_table(paris.aggregate(fcntx, "QE"))


# ------------------------------------------------------------------ weights
def test_weights_must_sum_to_one_within_1e4(funds):
    with pytest.raises(ParisError, match="sum"):
        paris.portfolio_return(funds, [0.3, 0.2, 0.2, 0.1, 0.1, 0.1002])
    assert paris.portfolio_return(funds, [0.3, 0.2, 0.2, 0.1, 0.1, 0.10009]).notna().all()
    assert np.allclose(
        paris.portfolio_return(funds, [3, 2, 2, 1, 1, 1], normalize=True).to_numpy(),
        paris.portfolio_return(funds, W6).to_numpy(),
    )
    with pytest.raises(ParisError, match="sum"):
        paris.volatility_contribution(funds, [0.5] * 6)
    with pytest.raises(ParisError, match="sum"):
        paris.portfolio_return(funds, [0.5, -0.5, 0, 0, 0, 0], normalize=True)  # zero-sum row


def test_weight_names_and_counts_are_checked(funds):
    with pytest.raises(AlignmentError):
        paris.portfolio_return(funds, {"FCNTX": 0.5, "ZZZZ": 0.5})
    with pytest.raises(ValueError):
        paris.portfolio_return(funds, [0.5, 0.5])  # 2 weights for 6 assets, unnamed
    with pytest.raises(ParisError):
        paris.portfolio_return(funds, [0.3, np.nan, 0.2, 0.1, 0.1, 0.1])


def test_dated_weights_rules(funds, w_table):
    late = pd.DataFrame([W6], columns=FUNDS, index=[funds.index[-1] + pd.DateOffset(months=1)])
    with pytest.raises(AlignmentError):  # dated after the last return
        paris.portfolio_return(funds, late)
    with pytest.raises(ValueError, match="rebalance"):  # rebalance needs a one-time vector
        paris.portfolio_return(funds, w_table, rebalance="QE")
    with pytest.raises(AlignmentError):  # unsorted weight dates
        paris.portfolio_return(funds, w_table.iloc[::-1])
    with pytest.raises(ParisError, match="single weight vector"):  # risk budgeting takes one vector
        paris.volatility_contribution(funds, w_table)
    with pytest.raises(ParisError, match="single weight vector"):
        paris.cvar_contribution(funds, w_table)


def test_brinson_categories_and_dates_must_align(sleeves, style, funds):
    later = style.set_axis(style.index + pd.DateOffset(years=20))
    with pytest.raises(AlignmentError, match="common window"):
        paris.active_contribution(sleeves, [0.6, 0.4], later, [0.5, 0.5])
    with pytest.raises(AlignmentError):
        paris.active_contribution(sleeves, [0.6, 0.4], paris.aggregate(style, "QE"), [0.5, 0.5])


# ------------------------------------------------------------------ switch values
@pytest.mark.parametrize(
    "call",
    [
        lambda r, b: paris.sharpe(r, risk="es"),
        lambda r, b: paris.probabilistic_sharpe(r, method="exact"),
        lambda r, b: paris.volatility_skewness(r, stat="skew"),
        lambda r, b: paris.downside_deviation(r, method="bogus"),
        lambda r, b: paris.upside_risk(r, stat="bogus"),
        lambda r, b: paris.skewness(r, method="bogus"),
        lambda r, b: paris.kurtosis(r, method="bogus"),
        lambda r, b: paris.var(r, method="bogus"),
        lambda r, b: paris.cvar(r, method="bogus"),
        lambda r, b: paris.cagr(r, method="years"),
        lambda r, b: paris.period_returns(r, windows=["2Y"]),
        lambda r, b: paris.appraisal_ratio(r, b, method="bogus"),
        lambda r, b: paris.rolling(r, paris.sharpe, 1),
        lambda r, b: paris.rolling(r, paris.sharpe, len(r) + 1),
        lambda r, b: paris.beta(r, None),
    ],
    ids=[
        "sharpe.risk",
        "psr.method",
        "volskew.stat",
        "dd.method",
        "upside.stat",
        "skew.method",
        "kurt.method",
        "var.method",
        "cvar.method",
        "cagr.method",
        "period_returns.windows",
        "appraisal.method",
        "rolling.window1",
        "rolling.window_too_long",
        "beta.no_benchmark",
    ],
)
def test_unknown_switch_value_raises_value_error(call, fcntx, spx):
    with pytest.raises(ValueError):
        call(fcntx, spx)


def test_budgeting_and_brinson_switches(funds, sleeves, style):
    with pytest.raises(ValueError, match="method"):
        paris.var_contribution(funds, W6, method="historical")  # no Euler decomposition
    with pytest.raises(ValueError, match="method"):
        paris.cvar_contribution(funds, W6, method="gaussian_tail")
    with pytest.raises(ValueError):
        paris.brinson(sleeves, [0.6, 0.4], style, [0.5, 0.5], method="BHX")
    with pytest.raises(ValueError):
        paris.brinson(sleeves, [0.6, 0.4], style, [0.5, 0.5], linking="carino2")


def test_errors_are_value_errors_too(fcntx):
    assert issubclass(ParisError, ValueError)
    for cls in (GapError, FrequencyError, AlignmentError):
        assert issubclass(cls, ParisError)
    holed = fcntx.copy()
    holed.iloc[10] = np.nan
    with pytest.raises(ValueError):
        paris.sharpe(holed)
