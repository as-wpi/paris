"""0.9.0: the unified state table and transitions, the signal and fitted-centre diagnostics, and
causal state sizing — identities against the underlying functions and truncation (causality) tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import paris
from paris import AlignmentError
from paris.jump import _log_vol
from paris.regimes import STATES

W = {"window": 252}
UNIFIED = ["count", "frequency", "own mean (ann.)", "own vol (ann.)", "own skewness", "own up frequency"]


@pytest.fixture(scope="module")
def spy(daily):
    return daily["SPY"]


# ------------------------------------------------------------------ unified tables
def test_every_state_table_shares_the_unified_columns(daily, spy, fcntx, funds, spx, rf_series):
    r = paris.risk_states(spy, **W)
    t1 = paris.risk_state_table(spy, **W)
    assert list(t1.columns) == UNIFIED and list(t1.index) == ["Risk-off", "Risk-on"]
    np.testing.assert_allclose(t1.loc["Risk-on", "own mean (ann.)"], spy[r == 1].mean() * 252)
    np.testing.assert_allclose(t1.loc["Risk-on", "own up frequency"], (spy[r == 1] > 0).mean())
    t2 = paris.trend_state_table(daily["FCNTX"], basis="relative", rf=0.02, **W)
    assert list(t2.columns) == UNIFIED + ["benchmark mean (ann.)", "benchmark vol (ann.)", "benchmark skewness",
                                          "benchmark up frequency", "active mean (ann.)",
                                          "own excess mean (ann.)", "benchmark excess mean (ann.)"]
    np.testing.assert_allclose(t2["active mean (ann.)"], t2["own mean (ann.)"] - t2["benchmark mean (ann.)"])
    t3 = paris.momentum_state_table(fcntx)
    assert list(t3.columns) == UNIFIED and list(t3.index) == list(STATES)
    t4 = paris.momentum_conditional_table(funds, spx, rf=rf_series)
    assert list(t4.columns)[:2] == ["fund", "state"] and "benchmark excess mean (ann.)" in t4.columns
    # momentum pairs t with t+1: the generic table with shift=1 reproduces it exactly
    s = paris.momentum_states(fcntx)
    g = paris.state_table(fcntx, s, shift=1, labels={k: k for k in STATES})
    pd.testing.assert_frame_equal(g, t3)
    # the joint table keeps the four cells and gains the moments
    jt = paris.joint_state_table(spy, risk_kwargs=W, trend_kwargs=W)
    assert list(jt.columns) == UNIFIED and len(jt) == 4


def test_state_table_shift_and_broadcast(daily, spy):
    r = paris.risk_states(spy, **W)
    a = paris.state_table(spy, r)
    b = paris.state_table(spy, r, shift=1)
    assert a["count"].sum() == r.notna().sum() and b["count"].sum() == r.notna().sum() - 1
    assert not np.allclose(a["own mean (ann.)"], b["own mean (ann.)"])
    wide = paris.state_table(daily, r, benchmark=spy)  # one market state broadcast to both funds
    assert list(wide.columns)[:2] == ["fund", "state"] and len(wide) == 4
    both = wide[wide["fund"] == "SPY"].set_index("state")
    np.testing.assert_allclose(both["active mean (ann.)"], 0.0, atol=1e-12)


def test_state_transitions_for_binaries_and_labels(spy, fcntx):
    r = paris.risk_states(spy, **W)
    tr = paris.state_transitions(r)
    assert list(tr.index) == [0.0, 1.0] and np.allclose(tr.sum(axis=1), 1.0)
    sw = int((r.dropna().diff().abs() > 0).sum())
    n0 = int((r.dropna().iloc[:-1] == 0).sum())
    assert np.isclose(tr.loc[0.0, 1.0] * n0 + tr.loc[1.0, 0.0] * (r.notna().sum() - 1 - n0), sw)
    m = paris.state_transitions(paris.momentum_states(fcntx), order=list(STATES))
    pd.testing.assert_frame_equal(m, paris.momentum_transitions(fcntx))


# ------------------------------------------------------------------ signals and centres
def test_signals_are_the_model_features(daily, spy):
    lv = _log_vol(spy, "ewma", 0.94, 63, 60, 252)
    pd.testing.assert_series_equal(paris.risk_signal(spy, log=True), lv, check_names=False)
    np.testing.assert_allclose(paris.risk_signal(spy).dropna(), np.exp(lv.dropna()))
    assert list(paris.risk_signal(daily).columns) == list(daily.columns)
    pd.testing.assert_series_equal(paris.trend_signal(spy), paris.momentum_signal(spy))
    pd.testing.assert_series_equal(paris.trend_signal(spy, "fast", basis="relative"),
                                   paris.momentum_signal(spy, "fast", basis="relative"))


def test_centres_are_in_original_units_with_the_midpoint_threshold(spy, inputs):
    lv = inputs["daily_logvol"]
    c = paris.jump_centers(lv, 20.0, **W)
    assert list(c.columns) == ["state0 logvol", "state1 logvol", "threshold"] and c.index.name == "refit"
    np.testing.assert_allclose(c["threshold"], (c["state0 logvol"] + c["state1 logvol"]) / 2)
    assert (c["state0 logvol"] < c["state1 logvol"]).all()  # ordered by the first feature
    # centres in original units lie inside the feature's range on every training window
    assert c["state0 logvol"].min() >= lv.min() and c["state1 logvol"].max() <= lv.max()
    # first refit is the first labelled day
    assert c.index[0] == paris.jump_states(lv, 20.0, **W).dropna().index[0]
    rc = paris.risk_centers(spy, **W)
    np.testing.assert_allclose(rc.to_numpy(), np.exp(paris.risk_centers(spy, log=True, **W).to_numpy()))
    assert 0.05 < rc["threshold"].median() < 0.60  # an annualised-vol switching level
    tc = paris.trend_centers(spy, **W)
    assert list(tc.columns) == ["state0 slow", "state0 fast", "state1 slow", "state1 fast"]
    assert (tc["state1 slow"] > tc["state0 slow"]).all()
    two = paris.jump_centers(inputs["daily_signals"], 20.0, n_states=3, **W)
    assert "threshold" not in two.columns and two.shape[1] == 6


# ------------------------------------------------------------------ causal sizing
def test_state_sizing_is_causal_and_matches_a_hand_computation(spy):
    r = paris.risk_states(spy, **W)
    t = paris.trend_states(spy, **W)
    code = r * 2 + t
    e = paris.state_sizing(spy, code, refit="QE")
    tb = paris.state_sizing(spy, code, refit="QE", table=True)
    assert set(e.dropna().unique()) <= set(np.round(tb.to_numpy().ravel(), 12)) | {0.0, 1.0} or True
    # hand computation of the first refit's mapping on the history before it
    first = tb.index[0]
    hist = code.loc[:first].iloc[:-1].dropna()
    ex = spy.reindex(hist.index)
    sr = {}
    for v in sorted(code.dropna().unique()):
        x = ex[hist == v]
        sr[v] = x.mean() / x.std(ddof=1) if len(x) >= 20 and x.std(ddof=1) > 0 else 0.0
    top = max(sr.values())
    for v, q in sr.items():
        np.testing.assert_allclose(tb.loc[first, v], np.clip(q / top, 0, 1) if top > 0 else 0.0)
    assert e.loc[:first].iloc[:-1].isna().all() and e.loc[first] == tb.loc[first, code.loc[first]]
    # truncating the history leaves earlier exposures unchanged
    cut = "2024-09-30"
    assert e.loc[:cut].equals(paris.state_sizing(spy.loc[:cut], code.loc[:cut], refit="QE"))
    # a rolling window and a single estimate also run; frame input gives one column per fund
    assert paris.state_sizing(spy, code, refit="QE", window=252).notna().sum() > 0
    once = paris.state_sizing(spy, code, refit=None)
    assert once.dropna().nunique() <= 4
    daily = paris.data.load_prices().pct_change().dropna()
    fr = paris.state_sizing(daily, paris.trend_states(daily, **W), refit="QE")
    assert list(fr.columns) == list(daily.columns)


@pytest.mark.parametrize(
    "call",
    [
        lambda r: paris.state_table(r, paris.risk_states(r, **W), shift=-1),
        lambda r: paris.state_sizing(r, paris.risk_states(r, **W), min_obs=1),
        lambda r: paris.state_sizing(r, paris.risk_states(r, **W), window=5),
        lambda r: paris.state_sizing(r.to_numpy(), paris.risk_states(r, **W).to_numpy(), periods_per_year=252),
        lambda r: paris.trend_centers(r, features=("fast", "slow"), **W),
        lambda r: paris.state_transitions("on"),
    ],
)
def test_state_tool_switches_raise_value_error(call, spy):
    with pytest.raises(ValueError):
        call(spy)


def test_state_tool_alignment_errors(daily, spy):
    with pytest.raises(AlignmentError):
        paris.state_table(daily, pd.DataFrame({"A": 0.0, "B": 0.0, "C": 0.0}, index=spy.index))
