"""Shared inputs for the test suite: every fixture is derived from the frozen sample data that
ships in ``paris.data``; nothing is synthetic and nothing is fetched."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import paris

FUNDS = ["FCNTX", "AGTHX", "FMAGX", "AMCPX", "DODGX", "PRGFX"]
BENCHMARKS = ["SPY", "IWF", "IWD"]
RF_SCALAR = 0.02  # annual rate; de-annualised by the library
W6 = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]  # the README's one-time weight vector
W_SLEEVES, W_STYLE = [0.6, 0.4], [0.5, 0.5]  # the README's Brinson weights


def build_inputs() -> dict[str, Any]:
    """The named inputs the golden cases refer to (``In("funds")`` etc.). Built once per session
    by the fixtures below and once per run by ``generate_expected``; both must agree, so this is
    the only place they are defined."""
    m = paris.data.load_managers()
    funds = m[FUNDS]
    daily = paris.data.load_prices().pct_change().dropna()
    w_table = pd.DataFrame(
        [W6, [1 / 6] * 6], columns=FUNDS, index=pd.to_datetime(["2010-01-31", "2018-06-30"])
    )
    sleeves = funds[["FCNTX", "DODGX"]].set_axis(["Growth", "Value"], axis=1)
    style = m[["IWF", "IWD"]].set_axis(["Growth", "Value"], axis=1)
    return {
        "m": m,
        "funds": funds,
        "fcntx": m["FCNTX"],
        "spx": m["SPY"],
        "iwf": m["IWF"],
        "iwd": m["IWD"],
        "rf_series": m["TBILL3M"],
        "daily": daily,
        "daily_fcntx": daily["FCNTX"],
        "daily_spx": daily["SPY"],
        "w_table": w_table,
        "sleeves": sleeves,
        "style": style,
        "contrib": paris.contribution(funds, W6),
        "states_fcntx": paris.momentum_states(m["FCNTX"]),
        "states_funds": paris.momentum_states(funds),
    }


@pytest.fixture(scope="session")
def inputs() -> dict[str, Any]:
    return build_inputs()


@pytest.fixture(scope="session")
def m(inputs) -> pd.DataFrame:
    return inputs["m"]


@pytest.fixture(scope="session")
def funds(inputs) -> pd.DataFrame:
    return inputs["funds"]


@pytest.fixture(scope="session")
def fcntx(inputs) -> pd.Series:
    return inputs["fcntx"]


@pytest.fixture(scope="session")
def spx(inputs) -> pd.Series:
    return inputs["spx"]


@pytest.fixture(scope="session")
def rf_series(inputs) -> pd.Series:
    return inputs["rf_series"]


@pytest.fixture(scope="session")
def daily(inputs) -> pd.DataFrame:
    return inputs["daily"]


@pytest.fixture(scope="session")
def w_table(inputs) -> pd.DataFrame:
    return inputs["w_table"]


@pytest.fixture(scope="session")
def sleeves(inputs) -> pd.DataFrame:
    return inputs["sleeves"]


@pytest.fixture(scope="session")
def style(inputs) -> pd.DataFrame:
    return inputs["style"]
