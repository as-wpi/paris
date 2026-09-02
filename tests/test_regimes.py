"""Momentum turning-point regimes: analytical checks on constructed series, the strategy-weight
identities of the paper, the closed-form dynamic speeds against a hand computation, and the
error paths. The published US-market figures are reproduced in the reference manual from the
Fama-French factor file, which does not ship with the package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import paris
from paris import AlignmentError
from paris.regimes import STATES

PPY = {"periods_per_year": 12}


def _monthly(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-31", periods=len(values), freq="ME")
    return pd.Series(values, index=idx, dtype=float)


# ------------------------------------------------------------------ classification
def test_states_follow_the_sign_table():
    # 12 warm-up months of +1 %, then the four quadrants with slow=3, fast=1
    r = _monthly([0.01] * 3 + [0.02, -0.005, -0.03, -0.03, 0.01, -0.001, 0.0])
    s = paris.momentum_states(r, slow=3, fast=1)
    assert s.iloc[:2].isna().all()
    # t=2: mean(1,1,1)%>=0, last +1% -> Bull; t=3: mean(1,1,2)>=0, 2% -> Bull
    # t=4: mean(1,2,-0.5)>=0, -0.5% -> Correction; t=5: mean(2,-0.5,-3)<0, -3% -> Bear
    # t=6: mean(-0.5,-3,-3)<0, -3% -> Bear; t=7: mean(-3,-3,1)<0, +1% -> Rebound
    # t=8: mean(-3,1,-0.1)<0, -0.1% -> Bear; t=9: mean(1,-0.1,0)>=0, 0.0 (tie) -> Bull
    assert list(s.iloc[2:]) == ["Bull", "Bull", "Correction", "Bear", "Bear", "Rebound", "Bear", "Bull"]
    codes = paris.momentum_states(r, slow=3, fast=1, codes=True)
    assert list(codes.iloc[:2]) == [-1, -1] and list(codes.iloc[2:]) == [0, 0, 1, 2, 2, 3, 2, 0]
    assert paris.momentum_state(r, slow=3, fast=1) == "Bull"
    assert paris.momentum_state_age(r, slow=3, fast=1) == 1.0
    assert paris.momentum_state_age(r.iloc[:7], slow=3, fast=1) == 2.0  # two Bears in a row


def test_ties_are_nonnegative_and_warmup_is_the_slow_window(fcntx):
    zeros = pd.Series(0.0, index=fcntx.index[:24])
    s = paris.momentum_states(zeros)
    assert s.iloc[:11].isna().all() and (s.iloc[11:] == "Bull").all()
    assert (paris.momentum_speed_weights(zeros).iloc[11:] == 1.0).all()


def test_default_lookbacks_follow_the_frequency(fcntx, daily):
    assert paris.momentum_states(fcntx).isna().sum() == 11
    assert paris.momentum_states(daily["SPY"]).isna().sum() == 251
    weekly = paris.aggregate(daily["SPY"], "W-FRI")
    assert paris.momentum_states(weekly).isna().sum() == 51


def test_bases_use_the_right_signal_series(fcntx, spx, rf_series):
    raw = paris.momentum_signal(fcntx)
    ex = paris.momentum_signal(fcntx, basis="excess", rf=rf_series)
    rel = paris.momentum_signal(fcntx, basis="relative", benchmark=spx)
    np.testing.assert_allclose(ex.dropna(), (fcntx - rf_series).rolling(12).mean().dropna())
    np.testing.assert_allclose(rel.dropna(), (fcntx - spx).rolling(12).mean().dropna())
    assert not np.allclose(raw.dropna(), ex.dropna())
    # the bundled S&P 500 proxy is the default relative benchmark
    pd.testing.assert_series_equal(
        paris.momentum_states(fcntx, basis="relative"),
        paris.momentum_states(fcntx, basis="relative", benchmark=spx),
    )


def test_compound_and_arithmetic_agree_away_from_zero(fcntx):
    a = paris.momentum_signal(fcntx)
    c = paris.momentum_signal(fcntx, compound=True)
    np.testing.assert_allclose(c.dropna(), (1 + fcntx).rolling(12).apply(np.prod, raw=True).dropna() - 1)
    sa, sc = paris.momentum_states(fcntx), paris.momentum_states(fcntx, compound=True)
    disagree = (sa != sc) & sa.notna()
    assert disagree.mean() < 0.1  # signs differ only when the 12-month mean is near zero
    assert (a[disagree].abs() < 0.005).all()


# ------------------------------------------------------------------ conditional tables
def test_state_table_and_transitions_are_consistent(spx):
    t = paris.momentum_state_table(spx)
    assert list(t.index) == list(STATES)
    assert np.isclose(t["frequency"].sum(), 1.0) and t["count"].sum() == len(spx) - 12
    tr = paris.momentum_transitions(spx)
    rows = tr.dropna(how="all")
    np.testing.assert_allclose(rows.sum(axis=1), 1.0)
    assert (tr.loc["Bull", ["Bear", "Rebound"]].sum() + tr.loc["Bear", ["Bull", "Correction"]].sum()) < 0.5


def test_frame_tables_are_long_with_a_fund_column(funds):
    t = paris.momentum_state_table(funds)
    assert list(t.columns)[:2] == ["fund", "state"] and len(t) == 4 * funds.shape[1]
    tr = paris.momentum_transitions(funds)
    assert list(tr.columns)[:2] == ["fund", "from"] and len(tr) == 4 * funds.shape[1]


# ------------------------------------------------------------------ speeds
def test_speed_weight_identities(fcntx):
    s = paris.momentum_states(fcntx)
    for a in (0.0, 0.25, 0.5, 1.0):
        w = paris.momentum_speed_weights(fcntx, a=a)
        assert (w[s == "Bull"] == 1.0).all() and (w[s == "Bear"] == -1.0).all()
        np.testing.assert_allclose(w[s == "Correction"], 1 - 2 * a)
        np.testing.assert_allclose(w[s == "Rebound"], 2 * a - 1)
    slow = paris.momentum_speed_weights(fcntx, a=0.0)
    fast = paris.momentum_speed_weights(fcntx, a=1.0)
    med = paris.momentum_speed_weights(fcntx, a=0.5)
    np.testing.assert_allclose(med.dropna(), (0.5 * slow + 0.5 * fast).dropna())
    np.testing.assert_allclose(slow.dropna(), np.where(paris.momentum_signal(fcntx).dropna() >= 0, 1.0, -1.0))
    dyn = paris.momentum_speed_weights(fcntx, speeds={"Correction": 0.0, "Rebound": 1.0})
    assert (dyn.dropna() == 1.0).sum() == (s.isin(["Bull", "Correction", "Rebound"])).sum()


def test_dynamic_speeds_match_the_closed_form(spx):
    s, r = paris.momentum_states(spx).iloc[:-1], spx.iloc[1:]
    keep = s.notna().to_numpy()
    s, r = s[keep].to_numpy(), r[keep].to_numpy()
    n = len(s)
    m = lambda st, k=1: np.sum(r[s == st] ** k) / n  # noqa: E731  sample E[r^k | st] P[st]
    e = lambda st, k=1: np.mean(r[s == st] ** k)  # noqa: E731  sample E[r^k | st]
    kappa = (m("Bull", 2) + m("Bear", 2)) / (m("Bull") - m("Bear"))
    a_co = np.clip(0.5 * (1 - kappa * e("Correction") / e("Correction", 2)), 0, 1)
    a_re = np.clip(0.5 * (1 + kappa * e("Rebound") / e("Rebound", 2)), 0, 1)
    got = paris.dynamic_speeds(spx)
    assert list(got.index) == ["Correction", "Rebound"]
    np.testing.assert_allclose(got.to_numpy(), [a_co, a_re], rtol=1e-12)
    assert ((0 <= got) & (got <= 1)).all()


def test_dynamic_speeds_are_nan_when_bear_pays_more_than_bull():
    # Bear months followed by large gains violate E[r|Bull]P[Bull] > E[r|Bear]P[Bear]
    r = _monthly([0.01] * 12 + [-0.1, -0.1, 0.3, -0.1, -0.1, 0.3, 0.01, 0.01])
    assert paris.dynamic_speeds(r, slow=3, fast=1).isna().all()


# ------------------------------------------------------------------ shapes & errors
def test_frame_in_gives_one_column_per_fund(funds, fcntx):
    st = paris.momentum_states(funds)
    assert list(st.columns) == list(funds.columns)
    pd.testing.assert_series_equal(st["FCNTX"], paris.momentum_states(fcntx), check_names=False)
    assert list(paris.momentum_state(funds).index) == list(funds.columns)
    assert list(paris.dynamic_speeds(funds).columns) == list(funds.columns)


def test_portfolio_wrapper_prefills_rf_and_benchmark(funds, spx, rf_series):
    pf = paris.Portfolio(funds, benchmark=spx, rf=rf_series)
    pd.testing.assert_series_equal(pf.momentum_state(basis="excess"),
                                   paris.momentum_state(funds, basis="excess", rf=rf_series))
    pd.testing.assert_series_equal(pf.momentum_state(basis="relative"),
                                   paris.momentum_state(funds, basis="relative", benchmark=spx))


@pytest.mark.parametrize(
    "call",
    [
        lambda r: paris.momentum_states(r, basis="bogus"),
        lambda r: paris.momentum_states(r, basis="excess"),  # rf missing
        lambda r: paris.momentum_states(r, slow=1, fast=1),
        lambda r: paris.momentum_states(r, slow=12, fast=0),
        lambda r: paris.momentum_states(r, slow=6.0, fast=1),
        lambda r: paris.momentum_signal(r, signal="medium"),
        lambda r: paris.momentum_speed_weights(r, a=1.5),
        lambda r: paris.momentum_speed_weights(r, speeds={"Bull": 0.5}),
        lambda r: paris.momentum_speed_weights(r, speeds={"Rebound": 2.0}),
        lambda r: paris.momentum_states(paris.aggregate(r, "YE")),  # no default lookbacks
        lambda r: paris.momentum_states(paris.aggregate(r, "QE"), basis="relative"),  # no bundled SPY
    ],
)
def test_bad_switches_raise_value_error(call, fcntx):
    with pytest.raises(ValueError):
        call(fcntx)


def test_history_shorter_than_the_slow_window_is_an_alignment_error(fcntx):
    with pytest.raises(AlignmentError):
        paris.momentum_states(fcntx.iloc[:10])
    assert paris.momentum_states(fcntx.iloc[:12]).notna().sum() == 1
