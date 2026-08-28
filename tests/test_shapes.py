"""Input/output shapes: a Series in gives one number, a DataFrame in gives a Series indexed by
column in column order, and bare numpy arrays / lists are accepted with an explicit frequency."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import paris
from tests.cases import CASES, In
from tests.conftest import FUNDS

# the default case of every per-column scalar statistic (its frame result is a Series over FUNDS)
_inputs = None


def _scalar_cases():
    global _inputs
    from tests.conftest import build_inputs

    _inputs = _inputs or build_inputs()
    out = []
    for c in CASES:
        if not c.id.endswith("/default") or not any(a == In("funds") for a in c.args):
            continue
        if c.module == "budgeting":  # a Series over the assets, not one number per fund
            continue
        v = c.run(_inputs)
        if isinstance(v, pd.Series) and list(v.index) == FUNDS:
            out.append(c)
    return out


SCALAR_CASES = _scalar_cases()


def _swap(case, inputs, **repl):
    args = tuple(repl.get(a.key, inputs[a.key]) if isinstance(a, In) else a for a in case.args)
    kwargs = {
        k: (repl.get(v.key, inputs[v.key]) if isinstance(v, In) else v)
        for k, v in case.kwargs.items()
    }
    return args, kwargs


@pytest.mark.parametrize("case", SCALAR_CASES, ids=[c.fn for c in SCALAR_CASES])
def test_series_in_scalar_out_matches_the_frame_column(case, inputs, fcntx):
    fn = getattr(paris, case.fn)
    frame = fn(*_swap(case, inputs)[0], **_swap(case, inputs)[1])
    assert isinstance(frame, pd.Series) and list(frame.index) == FUNDS
    args, kwargs = _swap(case, inputs, funds=fcntx)
    one = fn(*args, **kwargs)
    assert isinstance(one, (float, int, np.floating, np.integer)), type(one)
    assert not isinstance(one, (pd.Series, pd.DataFrame))
    np.testing.assert_allclose(float(one), float(frame["FCNTX"]), rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("case", SCALAR_CASES, ids=[c.fn for c in SCALAR_CASES])
def test_numpy_and_list_inputs_give_the_same_number(case, inputs, fcntx, spx):
    fn = getattr(paris, case.fn)
    ref = fn(*_swap(case, inputs, funds=fcntx)[0], **_swap(case, inputs, funds=fcntx)[1])
    extra = (
        {"periods_per_year": 12} if "periods_per_year" in inspect.signature(fn).parameters else {}
    )
    args, kwargs = _swap(case, inputs, funds=fcntx.to_numpy(), spx=spx.to_numpy())
    np.testing.assert_allclose(
        float(fn(*args, **{**kwargs, **extra})), float(ref), rtol=1e-12, equal_nan=True
    )
    args, kwargs = _swap(case, inputs, funds=fcntx.tolist(), spx=spx.tolist())
    np.testing.assert_allclose(
        float(fn(*args, **{**kwargs, **extra})), float(ref), rtol=1e-12, equal_nan=True
    )


def test_frame_results_keep_column_order(funds, spx, rf_series):
    shuffled = funds[FUNDS[::-1]]
    assert list(paris.sharpe(shuffled, rf=rf_series).index) == FUNDS[::-1]
    assert list(paris.beta(shuffled, spx).index) == FUNDS[::-1]
    assert list(paris.stats(shuffled, spx).columns) == FUNDS[::-1] + ["SPY"]


def test_single_column_frame_is_still_a_frame_result(funds, spx):
    one = funds[["FCNTX"]]
    assert isinstance(paris.sharpe(one), pd.Series) and list(paris.sharpe(one).index) == ["FCNTX"]
    assert list(paris.stats(one, spx).columns) == ["FCNTX", "SPY"]
    assert list(paris.period_returns(one).columns) == ["FCNTX"]


def test_scalar_rf_is_annual_and_series_rf_is_per_period(fcntx, rf_series):
    # a constant per-period Series equal to the de-annualised scalar gives the same Sharpe
    per_period = (1 + 0.02) ** (1 / 12) - 1
    const = pd.Series(per_period, index=fcntx.index)
    assert np.isclose(paris.sharpe(fcntx, rf=0.02), paris.sharpe(fcntx, rf=const))
    assert np.isclose(
        paris.sharpe(fcntx, rf=0.02, compounding=False),
        paris.sharpe(fcntx, rf=pd.Series(0.02 / 12, index=fcntx.index)),
    )
    # the real T-bill series is per-period: annualising it lands near its mean level
    assert 0 < paris.sharpe(fcntx, rf=rf_series) < paris.sharpe(fcntx)


def test_drawdown_table_frame_has_a_leading_fund_column(funds, fcntx):
    long = paris.drawdown_table(funds, top=2)
    assert list(long.columns)[0] == "fund" and set(long["fund"]) == set(FUNDS)
    assert len(long) == 12
    assert "fund" not in paris.drawdown_table(fcntx).columns


def test_stats_registry_labels_are_the_table_rows(funds, spx):
    t = paris.stats(funds, spx)
    assert list(t.index) == list(paris.ABSOLUTE_METRICS) + list(paris.RELATIVE_METRICS)
    assert list(t.columns) == FUNDS + ["SPY"]
    assert paris.stats(funds).index.tolist() == list(paris.ABSOLUTE_METRICS)
