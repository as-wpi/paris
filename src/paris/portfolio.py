"""Thin object wrapper: ``Portfolio(returns, benchmark, rf).sharpe()`` etc. No statistics live here."""
from __future__ import annotations

from functools import partial
from typing import Any

from paris import attribution, budgeting, drawdown, ratios, relative, returns, risk, tables
from paris.summary import stats

_MODULES = (returns, risk, drawdown, ratios, relative, tables)


class Portfolio:
    """Bundle returns (Series/DataFrame), an optional benchmark and a risk-free rate, and expose
    every PARIS function as a method with those inputs pre-filled.

    >>> pf = Portfolio(rets, benchmark=spx, rf=0.02)
    >>> pf.sharpe(), pf.beta(), pf.drawdown_table(top=3), pf.calendar_table(), pf.stats()

    With ``weights`` the frame of asset returns is combined into one portfolio series first
    (:func:`paris.portfolio_return`); ``benchmark_weights`` does the same for a frame of benchmark
    assets. The asset frames stay available as ``assets`` / ``benchmark_assets`` and the
    contribution, risk-budgeting and Brinson functions become methods.

    >>> pf = Portfolio(funds, weights={"FCNTX": 0.6, "DODGX": 0.4}, benchmark=spx, rf=rf)
    >>> pf.sharpe(), pf.contribution(), pf.volatility_contribution(pct=True)
    """

    def __init__(self, returns: Any, benchmark: Any = None, rf: Any = 0.0,
                 periods_per_year: int | None = None, *, weights: Any = None,
                 benchmark_weights: Any = None, rebalance: str | None = None,
                 benchmark_rebalance: str | None = None, normalize: bool = False):
        self.assets = self.benchmark_assets = None
        self.weights, self.benchmark_weights = weights, benchmark_weights
        self.rebalance, self.benchmark_rebalance, self.normalize = rebalance, benchmark_rebalance, normalize
        if weights is not None:
            self.assets = returns
            returns = attribution.portfolio_return(returns, weights, rebalance=rebalance, normalize=normalize)
        if benchmark_weights is not None:
            if benchmark is None:
                raise ValueError("benchmark_weights needs a benchmark frame of asset returns")
            self.benchmark_assets = benchmark
            benchmark = attribution.portfolio_return(benchmark, benchmark_weights, rebalance=benchmark_rebalance,
                                                     normalize=normalize).rename("Benchmark")
        self.returns, self.benchmark, self.rf, self.periods_per_year = returns, benchmark, rf, periods_per_year

    def stats(self, **kwargs: Any):
        return stats(self.returns, self.benchmark, self.rf, self.periods_per_year, **kwargs)

    # ---- construction & attribution (need weights) -----------------------------------------
    def _own(self) -> dict[str, Any]:
        if self.weights is None:
            raise ValueError("this Portfolio was built without weights")
        return {"rebalance": self.rebalance, "normalize": self.normalize}

    def _pair(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        kw = self._own()
        if self.benchmark_weights is None:
            raise ValueError("this Portfolio was built without benchmark_weights")
        kw["benchmark_rebalance"] = self.benchmark_rebalance
        return (self.assets, self.weights, self.benchmark_assets, self.benchmark_weights), kw

    def contribution(self, **kwargs: Any):
        return attribution.contribution(self.assets, self.weights, **self._own(), **kwargs)

    def bop_weights(self, **kwargs: Any):
        return attribution.bop_weights(self.assets, self.weights, **self._own(), **kwargs)

    def eop_weights(self, **kwargs: Any):
        return attribution.eop_weights(self.assets, self.weights, **self._own(), **kwargs)

    def period_contributions(self, freq: str | None = None):
        return attribution.period_contributions(self.contribution(), freq)

    def active_contribution(self, **kwargs: Any):
        args, kw = self._pair()
        return attribution.active_contribution(*args, **kw, **kwargs)

    def brinson(self, **kwargs: Any):
        args, kw = self._pair()
        return attribution.brinson(*args, **kw, **kwargs)

    # ---- risk budgeting (one weight vector) --------------------------------------------------
    def _budget(self, fn, *args: Any, **kwargs: Any):
        self._own()
        return fn(self.assets, self.weights, *args, normalize=self.normalize, **kwargs)

    def volatility_contribution(self, **kwargs: Any):
        return self._budget(budgeting.volatility_contribution, periods_per_year=self.periods_per_year, **kwargs)

    def var_contribution(self, *args: Any, **kwargs: Any):
        return self._budget(budgeting.var_contribution, *args, **kwargs)

    def cvar_contribution(self, *args: Any, **kwargs: Any):
        return self._budget(budgeting.cvar_contribution, *args, **kwargs)

    def marginal_var(self, *args: Any, **kwargs: Any):
        return self._budget(budgeting.marginal_var, *args, **kwargs)

    def marginal_cvar(self, *args: Any, **kwargs: Any):
        return self._budget(budgeting.marginal_cvar, *args, **kwargs)

    # ---- everything else: the topic modules with returns / benchmark / rf pre-filled -----------
    def __getattr__(self, name: str):
        import inspect

        for mod in _MODULES:
            if name not in getattr(mod, "__all__", ()):
                continue  # public API only: internal helpers and _core re-exports are not methods
            fn = getattr(mod, name)
            params = inspect.signature(fn).parameters
            kw: dict[str, Any] = {}
            if "rf" in params:
                kw["rf"] = self.rf
            if "periods_per_year" in params:
                kw["periods_per_year"] = self.periods_per_year
            if next(iter(params)) == "benchmark":  # risk_premium: the benchmark IS the series
                return partial(fn, self.benchmark, **kw)
            if "benchmark" in params:  # by keyword: tables.rolling takes fn second
                return partial(fn, self.returns, benchmark=self.benchmark, **kw)
            return partial(fn, self.returns, **kw)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __dir__(self):
        names = set(super().__dir__())
        for mod in _MODULES:
            names.update(getattr(mod, "__all__", []))
        return sorted(names)
