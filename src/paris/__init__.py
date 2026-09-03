"""PARIS - Portfolio Analytics, Risk & Investment Statistics.

Performance statistics for a fund or a collection of funds, absolute and relative to a benchmark,
cross-checked against independent reference implementations and public portfolio tools.

>>> import paris
>>> paris.sharpe(returns), paris.max_drawdown(returns), paris.beta(returns, benchmark)
>>> paris.stats(returns, benchmark=benchmark, rf=0.02)
>>> paris.calendar_table(returns), paris.rolling(returns, paris.sharpe, 36)
>>> paris.portfolio_return(assets, weights), paris.brinson(assets, w, bench_assets, wb)

Disclaimer
----------
PARIS is a library for internal analytical, educational and research use only. It does not
provide investment, tax, legal or other professional advice, and its output must not be relied
upon for investment, trading, reporting or any other decision-making. The library is provided
"AS IS" without warranty of any kind (see the accompanying LICENSE file). Users are solely
responsible for independently verifying every number, method and convention before use. See
``DISCLAIMER.md`` for the full statement.
"""
from paris import (attribution, budgeting, data, drawdown, jump, ratios, regimes, relative, returns,
                   risk, tables)
from paris._core import AlignmentError, FrequencyError, GapError, ParisError
from paris.attribution import *
from paris.budgeting import *
from paris.drawdown import *
from paris.jump import *
from paris.portfolio import Portfolio
from paris.ratios import *
from paris.regimes import *
from paris.relative import *
from paris.returns import *
from paris.risk import *
from paris.summary import ABSOLUTE_METRICS, RELATIVE_METRICS, stats
from paris.tables import *

__version__ = "0.6.0"
__all__ = (
    ["ParisError", "GapError", "FrequencyError", "AlignmentError", "Portfolio", "stats",
     "ABSOLUTE_METRICS", "RELATIVE_METRICS", "data", "__version__"]
    + returns.__all__ + risk.__all__ + drawdown.__all__ + ratios.__all__ + relative.__all__
    + tables.__all__ + attribution.__all__ + budgeting.__all__ + regimes.__all__ + jump.__all__
)
