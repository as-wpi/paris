"""The registry of golden cases: one call per public function and per documented keyword
switch, on the sample data. ``test_golden.py`` runs every case and compares it with the value
frozen in ``tests/expected/<module>.json`` by ``generate_expected.py``.

``In("funds")`` names an input from ``conftest.build_inputs``; ``Fn("sharpe")`` names a public
function (for ``rolling``). Everything else is a literal. Case ids are ``<function>/<variant>``
and are the keys of the JSON files, so keep them stable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import paris


@dataclass(frozen=True)
class In:
    key: str


@dataclass(frozen=True)
class Fn:
    name: str


@dataclass(frozen=True)
class Case:
    id: str
    fn: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)

    @property
    def module(self) -> str:
        return getattr(paris, self.fn).__module__.rsplit(".", 1)[1]

    def resolve(self, inputs: dict[str, Any]) -> tuple[tuple, dict]:
        def r(v):
            if isinstance(v, In):
                return inputs[v.key]
            if isinstance(v, Fn):
                return getattr(paris, v.name)
            if isinstance(v, list):
                return [r(x) for x in v]
            if isinstance(v, dict):
                return {k: r(x) for k, x in v.items()}
            return v

        return tuple(r(a) for a in self.args), {k: r(v) for k, v in self.kwargs.items()}

    def run(self, inputs: dict[str, Any]) -> Any:
        args, kwargs = self.resolve(inputs)
        return getattr(paris, self.fn)(*args, **kwargs)


FUNDS, FCNTX, SPX, RF, DAILY = In("funds"), In("fcntx"), In("spx"), In("rf_series"), In("daily")
W6 = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
W6_RAW = [3, 2, 2, 1, 1, 1]  # normalize=True rescales this to W6
SLEEVES, STYLE = In("sleeves"), In("style")
WS, WB = [0.6, 0.4], [0.5, 0.5]

CASES: list[Case] = []


def _add(fn: str, variants: dict[str, tuple[tuple, dict]]) -> None:
    for variant, (args, kwargs) in variants.items():
        CASES.append(Case(f"{fn}/{variant}", fn, args, kwargs))


def _abs(fn: str, **extra: dict) -> None:
    """Absolute statistic: default on the fund frame, plus named keyword variants."""
    _add(fn, {"default": ((FUNDS,), {}), **{k: ((FUNDS,), v) for k, v in extra.items()}})


def _rel(fn: str, **extra: dict) -> None:
    """Benchmark-relative statistic: (funds, spx) plus keyword variants."""
    _add(fn, {"default": ((FUNDS, SPX), {}), **{k: ((FUNDS, SPX), v) for k, v in extra.items()}})


RF_BOTH = {"rf_series": {"rf": RF}, "rf_scalar": {"rf": 0.02}}

# ----------------------------------------------------------------------------- returns
_add("active_returns", {"default": ((FUNDS, SPX), {})})
_add(
    "aggregate",
    {
        "quarterly": ((FUNDS, "QE"), {}),
        "daily_to_monthly": ((DAILY, "ME"), {}),
        "yearly_arithmetic": ((FUNDS, "YE"), {"geometric": False}),
    },
)
_abs("annualized_return", arithmetic={"geometric": False})
_add("annualized_return", {"daily": ((DAILY,), {})})
_abs("avg_loss")
_abs("avg_return", include_zeros={"include_zeros": True})
_abs("avg_win")
_abs("best")
_add("best", {"daily_monthly": ((DAILY,), {"freq": "ME"})})
_abs("worst")
_add("worst", {"daily_monthly": ((DAILY,), {"freq": "ME"})})
_abs(
    "cagr",
    calendar={"method": "calendar"},
    calendar_365={"method": "calendar", "days_in_year": 365},
)
_add(
    "cagr",
    {
        "daily": ((DAILY,), {}),
        "daily_calendar": ((DAILY,), {"method": "calendar"}),
        "daily_calendar_start": ((DAILY,), {"method": "calendar", "start": "2020-12-31"}),
    },
)
_abs("calendar_returns", quarterly={"freq": "QE"}, arithmetic={"geometric": False})
_abs("consecutive_losses")
_abs("consecutive_wins")
_abs("cumulative_returns", arithmetic={"geometric": False})
_add("cumulative_returns", {"series": ((FCNTX,), {})})
_abs(
    "excess_returns",
    rf_series={"rf": RF},
    rf_scalar={"rf": 0.02},
    rf_scalar_simple={"rf": 0.02, "compounding": False},
    geometric={"rf": RF, "geometric": True},
)
_abs("period_returns", as_of={"as_of": "2024-06-30"}, windows={"windows": ["YTD", "1Y", "ITD"]})
_add("period_returns", {"daily": ((DAILY,), {}), "series": ((FCNTX,), {})})
_abs("total_return", arithmetic={"geometric": False})
_abs("wealth_index")
_add("wealth_index", {"series_start_100": ((FCNTX,), {"start": 100.0})})
_abs("win_rate", include_zeros={"include_zeros": True})

# ----------------------------------------------------------------------------- risk
_abs(
    "cvar",
    gaussian={"method": "gaussian"},
    modified={"method": "modified"},
    gaussian_tail={"method": "gaussian_tail"},
    c99={"confidence": 0.99},
    lower={"interpolation": "lower"},
    modified_non_operational={"method": "modified", "operational": False},
    gaussian_ddof1={"method": "gaussian", "ddof": 1},
)
_abs(
    "downside_deviation",
    mar={"mar": 0.005},
    subset={"method": "subset"},
    annualized={"annualize": True},
)
_abs("downside_frequency", mar={"mar": 0.005})
_abs("gain_deviation")
_abs("loss_deviation")
_abs(
    "kurtosis",
    moment={"method": "moment"},
    sample={"method": "sample"},
    sample_excess={"method": "sample_excess"},
    fisher={"method": "fisher"},
)
_abs("mean_absolute_deviation")
_abs("outlier_loss_ratio", q05={"quantile": 0.05})
_abs("outlier_win_ratio", q95={"quantile": 0.95})
_abs("semi_deviation")
_abs("semi_variance")
_abs("skewness", fisher={"method": "fisher"}, sample={"method": "sample"})
_abs("tail_ratio", cutoff90={"cutoff": 0.9})
_abs("upside_frequency", mar={"mar": 0.005})
_abs(
    "upside_risk",
    subset={"method": "subset"},
    variance={"stat": "variance"},
    potential={"stat": "potential"},
)
_abs(
    "var",
    gaussian={"method": "gaussian"},
    modified={"method": "modified"},
    c99={"confidence": 0.99},
    higher={"interpolation": "higher"},
    gaussian_ddof1={"method": "gaussian", "ddof": 1},
)
_abs("volatility", per_period={"annualize": False}, ddof0={"ddof": 0})
_add("volatility", {"daily": ((DAILY,), {})})

# ----------------------------------------------------------------------------- drawdown
_abs("avg_drawdown", arithmetic={"geometric": False})
_abs("avg_drawdown_length")
_abs("avg_recovery")
_abs("burke_ratio", modified={"modified": True}, **RF_BOTH)
_abs("calmar_ratio", arithmetic={"geometric": False})
_abs("current_drawdown")
_abs("drawdown_deviation")
_add(
    "drawdown_table",
    {
        "series": ((FCNTX,), {}),
        "series_top3": ((FCNTX,), {"top": 3}),
        "frame_top2": ((FUNDS,), {"top": 2}),
    },
)
_abs("drawdowns")
_add("drawdowns", {"series_arithmetic": ((FCNTX,), {"geometric": False})})
_abs("longest_drawdown")
_abs("martin_ratio", ddof1={"ddof": 1}, **RF_BOTH)
_abs("max_drawdown", arithmetic={"geometric": False})
_abs("pain_index")
_abs("pain_ratio", **RF_BOTH)
_abs("recovery_factor")
_abs("sterling_ratio", excess05={"excess": 0.05})
_abs("ulcer_index", ddof1={"ddof": 1})
_abs("ulcer_performance_index", total={"annualize": False}, **RF_BOTH)

# ----------------------------------------------------------------------------- ratios
_abs("adjusted_sharpe", simple={"rf": 0.02, "compounding": False}, **RF_BOTH)
_abs("adjusted_sortino", mar={"mar": 0.005}, per_period={"annualize": False})
_abs("autocorr_penalty")
_abs("bernardo_ledoit_ratio")
_abs("common_sense_ratio")
_abs("cpc_index")
_abs("d_ratio")
_abs("gain_to_pain_ratio")
_add("gain_to_pain_ratio", {"daily_monthly": ((DAILY,), {"freq": "ME"})})
_abs("kappa", order2={"order": 2}, mar={"mar": 0.005})
_abs("kelly_criterion")
_abs("kelly_ratio", half={"half": True}, simple={"rf": 0.02, "compounding": False}, **RF_BOTH)
_abs("omega", mar={"mar": 0.005})
_abs("payoff_ratio")
_abs(
    "probabilistic_sharpe",
    moment={"method": "moment"},
    rf_scalar={"rf": 0.02},
    benchmark05={"benchmark_sharpe": 0.5},
    simple={"rf": 0.02, "compounding": False},
)
_abs("profit_factor")
_abs("prospect_ratio", mar={"mar": 0.005})
_abs("risk_of_ruin")
_abs("risk_return_ratio")
_abs("serenity_index", rf_scalar={"rf": 0.02}, ddof1={"ddof": 1})
_abs(
    "sharpe",
    per_period={"annualize": False},
    geometric={"rf": RF, "geometric": True},
    simple={"rf": 0.02, "compounding": False},
    smart={"smart": True},
    var={"risk": "var"},
    cvar={"risk": "cvar"},
    var_c99={"risk": "var", "confidence": 0.99},
    ddof0={"ddof": 0},
    **RF_BOTH,
)
_add("sharpe", {"daily": ((DAILY,), {"rf": 0.02})})
_abs("skewness_kurtosis_ratio")
_abs(
    "sortino",
    mar={"mar": 0.005},
    per_period={"annualize": False},
    subset={"method": "subset"},
    smart={"smart": True},
)
_abs("upside_potential_ratio", full={"method": "full"}, mar={"mar": 0.005})
_abs("volatility_skewness", variability={"stat": "variability"}, mar={"mar": 0.005})
_abs("win_loss_ratio")

# ----------------------------------------------------------------------------- relative
_rel("active_premium", arithmetic={"geometric": False})
_rel("alpha", per_period={"annualize": False}, arithmetic={"geometric": False}, **RF_BOTH)
_rel(
    "appraisal_ratio",
    modified={"method": "modified"},
    alternative={"method": "alternative"},
    rf_series={"rf": RF},
)
_rel("batting_average")
_rel("bear_beta", rf_series={"rf": RF})
_rel("bull_beta", rf_series={"rf": RF})
_rel("beta", **RF_BOTH)
_rel("capture_ratio", arithmetic={"geometric": False}, annualized={"annualize": True})
_rel("correlation")
_rel("down_capture", arithmetic={"geometric": False}, annualized={"annualize": True})
_rel("up_capture", arithmetic={"geometric": False}, annualized={"annualize": True})
_rel("down_number_ratio")
_rel("down_percentage_ratio")
_rel("up_number_ratio")
_rel("up_percentage_ratio")
_rel("fama_beta")
_rel("information_ratio", arithmetic={"geometric": False})
_rel("jensen_alpha", **RF_BOTH)
_rel("m_squared", **RF_BOTH)
_rel("m_squared_excess", rf_scalar={"rf": 0.02}, arithmetic={"geometric": False})
_rel("modigliani", rf_series={"rf": RF})
_rel("net_selectivity", rf_scalar={"rf": 0.02})
_rel("r_squared", rf_series={"rf": RF})
_rel("regression_stats", rf_series={"rf": RF})
_add(
    "risk_premium",
    {
        "benchmark": ((SPX,), {}),
        "benchmark_rf_scalar": ((SPX,), {"rf": 0.02}),
        "benchmark_rf_series": ((SPX,), {"rf": RF}),
        "funds": ((FUNDS,), {"rf": RF}),
    },
)
_rel("selectivity", rf_series={"rf": RF})
_rel("specific_risk", rf_series={"rf": RF})
_rel("systematic_risk", rf_series={"rf": RF})
_rel("timing_ratio", rf_series={"rf": RF})
_rel("total_risk", rf_series={"rf": RF})
_rel("tracking_error")
_add("tracking_error", {"daily": ((In("daily_fcntx"), In("daily_spx")), {})})
_rel("treynor_ratio", modified={"modified": True}, **RF_BOTH)

# ----------------------------------------------------------------------------- tables
_add(
    "annualized_table",
    {
        "default": ((FUNDS,), {}),
        "benchmark_rf": ((FUNDS, SPX), {"rf": RF}),
        "arithmetic": ((FUNDS,), {"geometric": False}),
    },
)
_add(
    "calendar_table",
    {
        "series": ((FCNTX,), {}),
        "series_benchmark": ((FCNTX, SPX), {}),
        "frame_benchmark": ((FUNDS, SPX), {}),
        "arithmetic": ((FCNTX, SPX), {"geometric": False}),
        "daily": ((In("daily_fcntx"),), {}),
    },
)
_add("capture_table", {"default": ((FUNDS, SPX), {})})
_add("distribution_table", {"default": ((FUNDS,), {}), "benchmark": ((FUNDS, SPX), {})})
_add(
    "downside_table",
    {
        "default": ((FUNDS,), {}),
        "benchmark_rf": ((FUNDS, SPX), {"rf": RF}),
        "mar_c99": ((FUNDS,), {"mar": 0.005, "confidence": 0.99}),
    },
)
_add(
    "drawdown_ratio_table", {"default": ((FUNDS,), {}), "benchmark_rf": ((FUNDS, SPX), {"rf": RF})}
)
_add("drawdown_summary", {"default": ((FUNDS,), {}), "benchmark": ((FUNDS, SPX), {})})
_add(
    "rolling",
    {
        "sharpe36": ((FUNDS, Fn("sharpe"), 36), {"rf": RF}),
        "beta36": ((FUNDS, Fn("beta"), 36, SPX), {}),
        "volatility12_series": ((FCNTX, Fn("volatility"), 12), {}),
        "tracking_error36_untrimmed": ((FUNDS, Fn("tracking_error"), 36, SPX), {"trim": False}),
        "sharpe36_kwargs": ((FUNDS, Fn("sharpe"), 36), {"rf": 0.02, "annualize": False}),
    },
)

# ----------------------------------------------------------------------------- attribution
_add(
    "active_contribution",
    {
        "default": ((SLEEVES, WS, STYLE, WB), {}),
        "rebalanced": ((SLEEVES, WS, STYLE, WB), {"rebalance": "QE", "benchmark_rebalance": "YE"}),
    },
)
for name in ("bop_weights", "eop_weights", "contribution"):
    _add(
        name,
        {
            "default": ((FUNDS, W6), {}),
            "quarterly": ((FUNDS, W6), {"rebalance": "QE"}),
            "dated": ((FUNDS, In("w_table")), {}),
            "normalized": ((FUNDS, W6_RAW), {"normalize": True}),
        },
    )
_add(
    "brinson",
    {
        "default": ((SLEEVES, WS, STYLE, WB), {}),
        "bhb": ((SLEEVES, WS, STYLE, WB), {"method": "BHB"}),
        "menchero": ((SLEEVES, WS, STYLE, WB), {"linking": "menchero"}),
        "arithmetic": ((SLEEVES, WS, STYLE, WB), {"linking": "none"}),
        "bhb_arithmetic": ((SLEEVES, WS, STYLE, WB), {"method": "BHB", "linking": "none"}),
        "yearly": ((SLEEVES, WS, STYLE, WB), {"freq": "YE"}),
        "rebalanced": ((SLEEVES, WS, STYLE, WB), {"rebalance": "QE", "benchmark_rebalance": "QE"}),
    },
)
_add(
    "period_contributions",
    {
        "default": ((In("contrib"),), {}),
        "quarterly": ((In("contrib"),), {"freq": "QE"}),
        "yearly": ((In("contrib"),), {"freq": "YE"}),
    },
)
_add(
    "portfolio_return",
    {
        "default": ((FUNDS, W6), {}),
        "quarterly": ((FUNDS, W6), {"rebalance": "QE"}),
        "yearly": ((FUNDS, W6), {"rebalance": "YE"}),
        "dated": ((FUNDS, In("w_table")), {}),
        "normalized": ((FUNDS, W6_RAW), {"normalize": True}),
        "subset_dict": ((FUNDS, {"FCNTX": 0.5, "DODGX": 0.5}), {}),
    },
)

# ----------------------------------------------------------------------------- budgeting
_add(
    "cvar_contribution",
    {
        "default": ((FUNDS, W6), {}),
        "gaussian": ((FUNDS, W6), {"method": "gaussian"}),
        "modified": ((FUNDS, W6), {"method": "modified"}),
        "pct": ((FUNDS, W6), {"pct": True}),
        "c99": ((FUNDS, W6), {"confidence": 0.99}),
        "modified_non_operational": ((FUNDS, W6), {"method": "modified", "operational": False}),
        "gaussian_ddof0": ((FUNDS, W6), {"method": "gaussian", "ddof": 0}),
        "lower": ((FUNDS, W6), {"interpolation": "lower"}),
    },
)
_add(
    "marginal_cvar",
    {
        "default": ((FUNDS, W6), {}),
        "gaussian": ((FUNDS, W6), {"method": "gaussian"}),
        "modified": ((FUNDS, W6), {"method": "modified"}),
    },
)
_add(
    "marginal_var",
    {
        "default": ((FUNDS, W6), {}),
        "gaussian": ((FUNDS, W6), {"method": "gaussian"}),
        "modified": ((FUNDS, W6), {"method": "modified"}),
    },
)
_add(
    "var_contribution",
    {
        "default": ((FUNDS, W6), {}),
        "modified": ((FUNDS, W6), {"method": "modified"}),
        "pct": ((FUNDS, W6), {"pct": True}),
        "c99": ((FUNDS, W6), {"confidence": 0.99}),
        "ddof0": ((FUNDS, W6), {"ddof": 0}),
    },
)
_add(
    "volatility_contribution",
    {
        "default": ((FUNDS, W6), {}),
        "pct": ((FUNDS, W6), {"pct": True}),
        "per_period": ((FUNDS, W6), {"annualize": False}),
        "ddof0": ((FUNDS, W6), {"ddof": 0}),
    },
)

# ----------------------------------------------------------------------------- summary
_add(
    "stats",
    {
        "default": ((FUNDS,), {}),
        "benchmark_rf": ((FUNDS, SPX), {"rf": RF}),
        "no_benchmark_column": ((FUNDS, SPX), {"rf": 0.02, "include_benchmark": False}),
        "series": ((FCNTX, SPX), {}),
        "selected": ((FUNDS, SPX), {"metrics": ["CAGR", "Sharpe", "Beta"]}),
    },
)

# ----------------------------------------------------------------------------- drawdown (0.6.0)
CAL = {"method": "calendar"}
CAL_START = {"method": "calendar", "start": "2009-12-31"}
_add(
    "calmar_ratio",
    {
        "window36": ((FUNDS,), {"window": 36}),
        "window36_arithmetic": ((FUNDS,), {"window": 36, "geometric": False}),
        "calendar": ((FUNDS,), CAL),
        "calendar_start": ((FUNDS,), CAL_START),
        "calendar_arithmetic": ((FUNDS,), {"method": "calendar", "geometric": False}),
        "daily_window252": ((DAILY,), {"window": 252}),
        "daily_calendar_start": ((DAILY,), {"method": "calendar", "start": "2020-12-31"}),
    },
)
_add(
    "sterling_ratio",
    {"window36": ((FUNDS,), {"window": 36}), "calendar": ((FUNDS,), CAL), "calendar_start": ((FUNDS,), CAL_START)},
)
_add(
    "martin_ratio",
    {
        "calendar": ((FUNDS,), CAL),
        "calendar_start": ((FUNDS,), CAL_START),
        "calendar_rf_series": ((FUNDS,), {"method": "calendar", "rf": RF}),
        "daily_calendar_start": ((DAILY,), {"method": "calendar", "start": "2020-12-31"}),
    },
)
_add("pain_ratio", {"calendar": ((FUNDS,), CAL), "calendar_start": ((FUNDS,), CAL_START)})
_add("burke_ratio", {"calendar": ((FUNDS,), CAL), "calendar_start_modified": ((FUNDS,), {**CAL_START, "modified": True})})
_add("ulcer_index", {"pct": ((FUNDS,), {"pct": True}), "pct_ddof1": ((FUNDS,), {"pct": True, "ddof": 1})})
_add(
    "rolling_ulcer",
    {
        "default": ((FUNDS, 12), {}),
        "daily252": ((DAILY, 252), {}),
        "series_pct": ((FCNTX, 36), {"pct": True}),
        "untrimmed": ((FCNTX, 36), {"trim": False}),
        "ddof1_arithmetic": ((FUNDS, 12), {"ddof": 1, "geometric": False}),
        "default_window": ((FUNDS,), {}),
        "daily_default_window": ((DAILY,), {}),
    },
)

_add(
    "drawdown_distribution",
    {
        "default": ((FUNDS,), {}),
        "series": ((FCNTX,), {}),
        "ulcer": ((FUNDS,), {"stat": "ulcer"}),
        "ulcer_window36": ((FUNDS,), {"stat": "ulcer", "window": 36}),
        "daily": ((DAILY,), {}),
        "daily_ulcer": ((DAILY,), {"stat": "ulcer"}),
        "quantiles_threshold": ((FUNDS,), {"quantiles": (0.5, 0.9), "threshold": 0.05}),
        "arithmetic": ((FUNDS,), {"geometric": False}),
    },
)

# ----------------------------------------------------------------------------- ratios (0.6.0)
_add(
    "kelly_ratio",
    {
        "fraction": ((FUNDS,), {"fraction": 0.25}),
        "half_fraction": ((FUNDS,), {"half": True, "fraction": 0.5}),
        "excess_var_rf_series": ((FUNDS,), {"excess_var": True, "rf": RF}),
    },
)
_abs("kelly_interval", c90={"confidence": 0.90}, half={"half": True}, excess_var={"excess_var": True}, **RF_BOTH)
_add("kelly_interval", {"series": ((FCNTX,), {"rf": RF}), "daily": ((DAILY,), {})})
_abs(
    "deflated_sharpe",
    trials10={"trials": 10, "sharpe_variance": 0.01},
    trial_sharpes={"trials": [0.05, 0.10, 0.20, 0.15, 0.30]},
    moment={"trials": 10, "sharpe_variance": 0.01, "method": "moment"},
    **RF_BOTH,
)
_abs(
    "min_track_record",
    benchmark05={"benchmark_sharpe": 0.05},
    years={"years": True},
    c99={"confidence": 0.99},
    moment={"method": "moment"},
    **RF_BOTH,
)
# ----------------------------------------------------------------------------- regimes
IWF = In("iwf")
_abs(
    "momentum_signal",
    fast={"signal": "fast"},
    compound={"compound": True},
    excess_rf_series={"basis": "excess", "rf": RF},
    relative_iwf={"basis": "relative", "benchmark": IWF},
    lookbacks_6_2={"slow": 6, "fast": 2},
)
_add("momentum_signal", {"daily": ((DAILY,), {}), "series": ((SPX,), {})})
_abs(
    "momentum_states",
    codes={"codes": True},
    compound={"compound": True},
    excess_rf_series={"basis": "excess", "rf": RF},
    excess_rf_scalar={"basis": "excess", "rf": 0.02},
    relative_default_spx={"basis": "relative"},
    relative_iwf={"basis": "relative", "benchmark": IWF},
    lookbacks_6_2={"slow": 6, "fast": 2},
)
_add(
    "momentum_states",
    {
        "series": ((SPX,), {}),
        "daily": ((DAILY,), {}),
        "daily_compound_codes": ((DAILY,), {"compound": True, "codes": True}),
        "daily_relative_default_spx": ((In("daily_fcntx"),), {"basis": "relative"}),
    },
)
_add(
    "momentum_state",
    {
        "frame": ((FUNDS,), {}),
        "series": ((FCNTX,), {}),
        "excess_rf_series": ((FUNDS,), {"basis": "excess", "rf": RF}),
        "relative_default_spx": ((FUNDS,), {"basis": "relative"}),
        "daily": ((DAILY,), {}),
    },
)
_abs("momentum_state_age", excess_rf_series={"basis": "excess", "rf": RF}, compound={"compound": True})
_add("momentum_state_age", {"series": ((FCNTX,), {}), "daily": ((DAILY,), {})})
_add(
    "momentum_state_table",
    {
        "default": ((FUNDS,), {}),
        "series": ((SPX,), {}),
        "excess_rf_series": ((SPX,), {"basis": "excess", "rf": RF}),
        "relative_iwf": ((FUNDS,), {"basis": "relative", "benchmark": IWF}),
        "daily": ((DAILY,), {}),
    },
)
_add(
    "momentum_transitions",
    {
        "default": ((FUNDS,), {}),
        "series": ((SPX,), {}),
        "excess_rf_series": ((SPX,), {"basis": "excess", "rf": RF}),
        "daily": ((DAILY,), {}),
    },
)
_add(
    "momentum_speed_weights",
    {
        "default": ((FUNDS,), {}),
        "slow": ((FUNDS,), {"a": 0.0}),
        "fast": ((FUNDS,), {"a": 1.0}),
        "speeds": ((FUNDS,), {"speeds": {"Correction": 0.0, "Rebound": 1.0}}),
        "series_excess": ((SPX,), {"basis": "excess", "rf": RF}),
        "daily": ((DAILY,), {"a": 0.75}),
    },
)
_add(
    "dynamic_speeds",
    {
        "default": ((FUNDS,), {}),
        "series": ((SPX,), {}),
        "excess_rf_series": ((SPX,), {"basis": "excess", "rf": RF}),
        "relative_default_spx": ((FUNDS,), {"basis": "relative"}),
        "daily": ((DAILY,), {}),
    },
)

_ids = [c.id for c in CASES]
assert len(_ids) == len(set(_ids)), "duplicate case ids"
BY_MODULE: dict[str, list[Case]] = {}
for _c in CASES:
    BY_MODULE.setdefault(_c.module, []).append(_c)
