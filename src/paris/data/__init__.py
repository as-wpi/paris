"""Sample datasets for illustrating PARIS.

Two frozen CSV files ship inside the package; the loaders only read them.

* :func:`load_managers` — monthly simple **total** returns, 2010-01-31 .. 2025-12-31 (192 rows):
  six widely held active US large-cap mutual funds (``FCNTX`` Fidelity Contrafund, ``AGTHX``
  American Funds Growth Fund of America, ``FMAGX`` Fidelity Magellan, ``AMCPX`` American Funds
  AMCAP, ``DODGX`` Dodge & Cox Stock, ``PRGFX`` T. Rowe Price Growth Stock), three total-return
  benchmark proxies (``SPY`` S&P 500, ``IWF`` Russell 1000 Growth, ``IWD`` Russell 1000 Value —
  ETFs, hence net of their expense ratios) and ``TBILL3M``, the 3-month Treasury bill yield at
  month-end converted to a per-month rate, ``(1 + y)**(1/12) - 1``, for use as a per-period ``rf``.
* :func:`load_prices` — daily total-return index levels of ``SPY`` and ``FCNTX``, trading days
  2021-01-04 .. 2025-12-31, rebased to the net asset value on 2020-12-31.

Total returns reinvest every distribution on its ex-date, ``r_t = (NAV_t + D_t)/NAV_{t-1} - 1``;
the NAV history is quoted to the cent, so returns carry a rounding granularity of roughly
0.01-0.05 % per observation. The frames are rectangular — no leading, trailing or interior
missing values — so every PARIS function accepts them whole. They are illustrative: they play no
part in the oracle tests and are not a benchmark-quality dataset. Frozen on 2026-08-21; see
the reference manual chapter 11 (Sample data) for provenance and the independent-data check.

>>> m = paris.data.load_managers()
>>> funds, spx, rf = m.iloc[:, :6], m["SPY"], m["TBILL3M"]
>>> paris.stats(funds, benchmark=spx, rf=rf)
"""
from __future__ import annotations

from importlib.resources import files

import pandas as pd

__all__ = ["describe", "load_managers", "load_prices"]

_MANAGERS = [
    ("FCNTX", "Fidelity Contrafund", "fund"),
    ("AGTHX", "American Funds Growth Fund of America", "fund"),
    ("FMAGX", "Fidelity Magellan", "fund"),
    ("AMCPX", "American Funds AMCAP", "fund"),
    ("DODGX", "Dodge & Cox Stock", "fund"),
    ("PRGFX", "T. Rowe Price Growth Stock", "fund"),
    ("SPY", "S&P 500 (ETF total-return proxy)", "benchmark"),
    ("IWF", "Russell 1000 Growth (ETF total-return proxy)", "benchmark"),
    ("IWD", "Russell 1000 Value (ETF total-return proxy)", "benchmark"),
    ("TBILL3M", "3-month Treasury bill yield, per month", "risk-free"),
]
_PRICES = [
    ("SPY", "S&P 500 (ETF total-return index level)", "benchmark"),
    ("FCNTX", "Fidelity Contrafund (total-return index level)", "fund"),
]
DATASETS = {
    "managers": ("managers.csv", "monthly simple total returns", "2010-01-31", "2025-12-31",
                 _MANAGERS),
    "prices": ("prices.csv", "daily total-return index levels", "2021-01-04", "2025-12-31",
               _PRICES),
}


def _read(name: str) -> pd.DataFrame:
    path = files("paris.data").joinpath(DATASETS[name][0])
    with path.open("r", encoding="utf-8") as fh:
        df = pd.read_csv(fh, index_col="date", parse_dates=True)
    df.index.name = "date"
    return df


def load_managers() -> pd.DataFrame:
    """Monthly total returns of six active US large-cap funds, SPY/IWF/IWD and TBILL3M, 2010-2025."""
    return _read("managers")


def load_prices() -> pd.DataFrame:
    """Daily total-return index levels of SPY and FCNTX, 2021-2025 (use ``.pct_change()`` for returns)."""
    return _read("prices")


def describe() -> pd.DataFrame:
    """One row per dataset column: dataset, column, name, role, frequency, first, last."""
    rows = [
        {"dataset": ds, "column": col, "name": name, "role": role, "frequency": freq,
         "first": first, "last": last}
        for ds, (_, freq, first, last, cols) in DATASETS.items()
        for col, name, role in cols
    ]
    return pd.DataFrame(rows)
