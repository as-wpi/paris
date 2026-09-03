"""Jump-model indicators: the in-house core against the reference ``jumpmodels`` package (oracle,
skipped when the ``oracle`` dependency group is not installed), causality of the rolling online
states, the lag and state-ordering conventions, the conditional tables and the error paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import paris
from paris import AlignmentError
from paris.jump import _fit, _log_vol, _loss, _online_all, _scaler, _standardise, _viterbi

W = {"window": 252}


@pytest.fixture(scope="module")
def spy(daily):
    return daily["SPY"]


@pytest.fixture(scope="module")
def logvol(spy):
    return _log_vol(spy, "ewma", 0.94, 63, 60, 252).dropna()


# ------------------------------------------------------------------ oracle
@pytest.mark.parametrize("lam", [1.0, 5.0, 20.0, 50.0])
def test_core_matches_the_reference_implementation(logvol, inputs, lam):
    jm_mod = pytest.importorskip("jumpmodels.jump")
    for feat in (logvol.to_frame(), inputs["daily_signals"]):
        X = feat.to_numpy(dtype=float)
        mu, sd = _scaler(X)
        Z = _standardise(X, mu, sd, 3.0)
        mine = _fit(Z, lam, 2, 10, np.random.default_rng(0))
        labels, val = _viterbi(_loss(Z, mine), lam)
        ref = jm_mod.JumpModel(2, jump_penalty=lam, cont=False, n_init=10, random_state=0).fit(Z, sort_by=None)
        # the reference seeds every init identically (fixed random_state), so it can stop in a
        # local optimum; ours must be at least as good, and identical when both reach the same one
        assert val <= ref.val_ * (1 + 1e-9), (val, ref.val_)
        if np.isclose(val, ref.val_, rtol=1e-9):
            np.testing.assert_allclose(mine, ref.centers_[np.argsort(ref.centers_[:, 0])], atol=1e-8)
        # with identical centres the hindsight path and the online path must agree exactly
        same = jm_mod.JumpModel(2, jump_penalty=lam)
        same.check_jump_penalty_mx()
        same.centers_ = mine
        np.testing.assert_array_equal(np.asarray(same.predict(Z)), labels)
        np.testing.assert_array_equal(np.asarray(same.predict_online(Z)), _online_all(_loss(Z, mine), lam))


def test_three_state_general_path_agrees_with_two_state_fast_path(logvol):
    # the K=2 scalar recursion and the general numpy DP must give the same answers
    X = logvol.to_numpy()[:, None]
    mu, sd = _scaler(X)
    Z = _standardise(X, mu, sd, 3.0)
    centers = np.array([[-1.0], [1.0]])
    loss = _loss(Z, centers)
    lab2, val2 = _viterbi(loss, 10.0)
    loss3 = np.column_stack([loss, np.full(len(loss), 1e6)])  # unreachable third state
    lab3, val3 = _viterbi(loss3, 10.0)
    np.testing.assert_array_equal(lab2, lab3)
    assert np.isclose(val2, val3)
    np.testing.assert_array_equal(_online_all(loss, 10.0), _online_all(loss3, 10.0))


# ------------------------------------------------------------------ causality & conventions
def test_rolling_states_are_causal(spy):
    full = paris.risk_states(spy, **W)
    for cut in ("2023-06-15", "2024-03-29", "2025-06-30"):
        assert full.loc[:cut].equals(paris.risk_states(spy.loc[:cut], **W)), cut
    fullt = paris.trend_states(spy, **W)
    assert fullt.loc[:"2024-06-28"].equals(paris.trend_states(spy.loc[:"2024-06-28"], **W))


def test_feature_weights_are_causal_and_effective(spy, logvol, inputs):
    # weights are applied inside every rolling window (no full-sample statistic): truncating the
    # history must leave earlier labels unchanged, exactly as without weights
    kw = dict(feature_weights=[1.0, 0.5], **W)
    full = paris.trend_states(spy, **kw)
    assert full.loc[:"2024-06-28"].equals(paris.trend_states(spy.loc[:"2024-06-28"], **kw))
    jw = dict(features=("logvol", "slow", "fast"), feature_weights=[1.0, 1.0, 0.25], **W)
    fj = paris.joint_states(spy, **jw)
    assert fj.loc[:"2024-06-28"].equals(paris.joint_states(spy.loc[:"2024-06-28"], **jw))
    # and they change the answer (a weight of 1 on every feature is the unweighted model)
    sig = inputs["daily_signals"]
    base = paris.jump_states(sig, 5.0, **W)
    assert base.equals(paris.jump_states(sig, 5.0, feature_weights=[1.0, 1.0], **W))
    assert not base.equals(paris.jump_states(sig, 5.0, feature_weights=[1.0, 0.1], **W))
    with pytest.raises(ValueError):
        paris.jump_states(sig, 5.0, feature_weights=[1.0], **W)
    with pytest.raises(ValueError):
        paris.jump_states(sig, 5.0, feature_weights=[1.0, -1.0], **W)


def test_lag_and_warmup(spy, logvol):
    lag0 = paris.risk_states(spy, lag=0, **W)
    lag1 = paris.risk_states(spy, **W)
    assert lag0.shift(1).equals(lag1)
    assert lag1.isna().sum() == 60 + 252 + 1  # vol warm-up + window + lag
    s = paris.jump_states(logvol, 20.0, **W)
    assert s.isna().sum() == 252 and set(s.dropna().unique()) <= {0.0, 1.0}


def test_state_ordering_conventions(spy, logvol):
    on = paris.risk_states(spy, lag=0, **W)
    assert logvol[on == 1].mean() < logvol[on == 0].mean()  # risk-on = low volatility
    t = paris.trend_states(spy, lag=0, **W)
    slow = paris.momentum_signal(spy)
    assert slow[t == 1].mean() > slow[t == 0].mean()  # trend-on = high slow signal
    lab = paris.jump_labels(logvol, 20.0)
    assert logvol[lab == 0].mean() < logvol[lab == 1].mean()


def test_benchmark_basis_broadcasts_the_market_state(daily, spy):
    rb = paris.risk_states(daily, basis="benchmark", **W)
    assert rb["SPY"].equals(rb["FCNTX"]) and rb["SPY"].equals(paris.risk_states(spy, **W))
    rb2 = paris.risk_states(daily, basis="benchmark", benchmark=spy, **W)
    pd.testing.assert_frame_equal(rb, rb2)


def test_rolling_and_ewma_features_differ_but_both_run(spy):
    a = paris.risk_states(spy, vol="ewma", **W)
    b = paris.risk_states(spy, vol="rolling", **W)
    assert a.dropna().mean() > 0.3 and b.dropna().mean() > 0.3
    assert not a.equals(b)


def test_single_fit_and_refit_agree_on_the_first_period(logvol):
    once = paris.jump_states(logvol, 20.0, window=252, refit=None)
    monthly = paris.jump_states(logvol, 20.0, window=252, refit="ME")
    first_month = monthly.dropna().index[0].to_period("M")
    idx = monthly.dropna().index
    same = idx[idx.to_period("M") == first_month]
    assert once[same].equals(monthly[same])


# ------------------------------------------------------------------ tables
def test_risk_table_ties_to_the_states(daily, spy):
    st = paris.risk_states(spy, **W)
    t = paris.risk_state_table(spy, **W)
    assert list(t.index) == ["Risk-off", "Risk-on"]
    assert t["count"].sum() == st.notna().sum() and np.isclose(t["frequency"].sum(), 1.0)
    np.testing.assert_allclose(t.loc["Risk-on", "own mean (ann.)"], spy[st == 1].mean() * 252)
    assert "benchmark mean (ann.)" not in t.columns
    tb = paris.risk_state_table(daily["FCNTX"], basis="benchmark", **W)
    assert "benchmark mean (ann.)" in tb.columns
    np.testing.assert_allclose(tb.loc["Risk-on", "benchmark mean (ann.)"], spy[st == 1].mean() * 252)
    long = paris.risk_state_table(daily, **W)
    assert list(long.columns)[:2] == ["fund", "state"] and len(long) == 4


def test_regime_runs_accepts_binary_states(spy):
    runs = paris.regime_runs(paris.risk_states(spy, **W))
    assert set(runs["state"]) <= {0.0, 1.0}
    assert runs["length"].sum() == paris.risk_states(spy, **W).notna().sum()


def test_trend_table_ties_to_the_states(daily, spy):
    st = paris.trend_states(daily["FCNTX"], basis="relative", **W)
    t = paris.trend_state_table(daily["FCNTX"], basis="relative", **W)
    assert list(t.index) == ["Trend-off", "Trend-on"]
    np.testing.assert_allclose(t.loc["Trend-on", "own mean (ann.)"], daily["FCNTX"][st == 1].mean() * 252)
    np.testing.assert_allclose(t.loc["Trend-on", "benchmark mean (ann.)"], spy[st == 1].mean() * 252)
    assert "benchmark mean (ann.)" not in paris.trend_state_table(spy, **W).columns


# ------------------------------------------------------------------ errors
@pytest.mark.parametrize(
    "call",
    [
        lambda r, f: paris.risk_states(r, basis="market", **W),
        lambda r, f: paris.risk_states(r, vol="garch", **W),
        lambda r, f: paris.risk_states(r, vol_lambda=1.0, **W),
        lambda r, f: paris.risk_states(r, lag=-1, **W),
        lambda r, f: paris.trend_states(r, features=("fast", "slow"), **W),
        lambda r, f: paris.trend_states(r, features=("slow", "slow"), **W),
        lambda r, f: paris.trend_states(r, feature_weights=[1.0], **W),
        lambda r, f: paris.jump_states(f, -1.0, **W),
        lambda r, f: paris.jump_states(f, 20.0, window=252, n_states=1),
        lambda r, f: paris.jump_states(f, 20.0, window=2),
        lambda r, f: paris.jump_states(f.to_numpy(), 20.0, window=252),  # refit needs dates
        lambda r, f: paris.jump_labels(f, 20.0, clip=0.0),
        lambda r, f: paris.regime_runs("Bull"),
    ],
)
def test_jump_switches_raise_value_error(call, spy, logvol):
    with pytest.raises(ValueError):
        call(spy, logvol)


def test_window_longer_than_history_is_an_alignment_error(spy, logvol):
    with pytest.raises(AlignmentError):
        paris.jump_states(logvol, 20.0, window=len(logvol))
    with pytest.raises(AlignmentError):
        paris.risk_states(spy, window=len(spy))
    assert isinstance(paris.jump_states(logvol.to_numpy(), 20.0, window=252, refit=None), pd.Series)


# ------------------------------------------------------------------ 0.8.0: combinations & joint models
def test_combine_states_identities(spy):
    r, t = paris.risk_states(spy, **W), paris.trend_states(spy, **W)
    g = paris.combine_states(r, t)
    assert g.isna().equals(r.isna() | t.isna()) and set(g.dropna().unique()) <= {0.0, 0.5, 1.0}
    pd.testing.assert_series_equal(g, (r + t) / 2, check_names=False)
    pd.testing.assert_series_equal(paris.combine_states(r, t, "gate"), t * (0.5 + 0.5 * r), check_names=False)
    pd.testing.assert_series_equal(paris.combine_states(r, t, "and"), r * t, check_names=False)
    pd.testing.assert_series_equal(paris.combine_states(r, t, "or"), np.maximum(r, t), check_names=False)
    cells = {(0, 0): 0.0, (0, 1): 1.0, (1, 0): 0.5, (1, 1): 1.0}
    c = paris.combine_states(r, t, "cells", cells=cells)
    m = (r == 1) & (t == 0)
    assert (c[m] == 0.5).all() and (c[(r == 0) & (t == 1)] == 1.0).all()
    # DataFrames with the same columns
    daily = paris.data.load_prices().pct_change().dropna()
    R, T = paris.risk_states(daily, **W), paris.trend_states(daily, **W)
    G = paris.combine_states(R, T)
    assert list(G.columns) == list(daily.columns)
    pd.testing.assert_series_equal(G["SPY"], g, check_names=False)


def test_state_table_and_joint_cells(daily, spy):
    r, t = paris.risk_states(spy, **W), paris.trend_states(spy, **W)
    st = paris.state_table(spy, r)
    assert list(st.index) == [0.0, 1.0] and st["count"].sum() == r.notna().sum()
    np.testing.assert_allclose(st.loc[1.0, "own mean (ann.)"], spy[r == 1].mean() * 252)
    lab = paris.state_table(spy, r, labels={0.0: "off", 1.0: "on"})
    assert list(lab.index) == ["off", "on"]
    jt = paris.joint_state_table(spy, risk_kwargs=W, trend_kwargs=W)
    assert list(jt.index) == ["Risk-off & Trend-off", "Risk-off & Trend-on", "Risk-on & Trend-off", "Risk-on & Trend-on"]
    assert jt["count"].sum() == (r.notna() & t.notna()).sum()
    both = (r == 1) & (t == 1)
    np.testing.assert_allclose(jt.loc["Risk-on & Trend-on", "own mean (ann.)"], spy[both].mean() * 252)
    jb = paris.joint_state_table(daily["FCNTX"], benchmark=spy, risk_kwargs=W, trend_kwargs=W)
    assert "benchmark mean (ann.)" in jb.columns
    long = paris.state_table(daily, paris.risk_states(daily, **W))
    assert list(long.columns)[:2] == ["fund", "state"] and len(long) == 4


def test_joint_states_conventions(spy):
    j2 = paris.joint_states(spy, **W)
    assert set(j2.dropna().unique()) <= {0.0, 1.0}
    lv = _log_vol(spy, "ewma", 0.94, 63, 60, 252)
    assert lv[j2 == 0].mean() < lv[j2 == 1].mean()  # logvol first: state 0 is the calm one
    j4 = paris.joint_states(spy, features=("logvol", "slow", "fast"), n_states=4, **W)
    assert set(j4.dropna().unique()) <= {0.0, 1.0, 2.0, 3.0}
    js = paris.joint_states(spy, features=("slow", "logvol"), lag=0, **W)
    slow = paris.momentum_signal(spy)
    assert slow[js == 1].mean() > slow[js == 0].mean()  # slow first: high state is the trend
    # causal like the binaries
    assert j2.loc[:"2024-06-28"].equals(paris.joint_states(spy.loc[:"2024-06-28"], **W))
    assert paris.joint_states(spy, lag=0, **W).shift(1).equals(j2)


@pytest.mark.parametrize(
    "call",
    [
        lambda r, f: paris.combine_states(paris.risk_states(r, **W), paris.trend_states(r, **W), "xor"),
        lambda r, f: paris.combine_states(paris.risk_states(r, **W), paris.trend_states(r, **W), "cells"),
        lambda r, f: paris.combine_states(paris.risk_states(r, **W), paris.trend_states(r, **W), "cells",
                                          cells={(0, 0): 0, (0, 1): 2, (1, 0): 0, (1, 1): 1}),
        lambda r, f: paris.joint_states(r, features=("logvol", "vix"), **W),
        lambda r, f: paris.joint_states(r, feature_weights=[1.0], **W),
        lambda r, f: paris.state_table(r, "on"),
    ],
)
def test_combination_switches_raise_value_error(call, spy, logvol):
    with pytest.raises(ValueError):
        call(spy, logvol)


def test_combination_alignment_errors(daily, spy):
    with pytest.raises(AlignmentError):
        paris.combine_states(paris.risk_states(daily, **W), paris.trend_states(spy, **W))
    with pytest.raises(AlignmentError):
        paris.state_table(daily, pd.DataFrame({"A": paris.risk_states(spy, **W), "B": 0.0, "C": 0.0}))
