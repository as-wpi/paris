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
