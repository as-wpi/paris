"""Trend / gate / graded signals computed on a 1x ETF, invested in its leveraged counterpart.

Fixed rules only (the walk-forward showed annual re-selection to be harmful): trend-on/off
(jump penalty 5, slow + fast features) and risk-on/off (penalty 50, log EWMA volatility), both
estimated on the 1x ETF with the library's rolling 1,260-day / monthly-refit / lag-1 machinery, so
the exposure for day T is known at the close of T-1. Exposure is applied to the leveraged fund's
day-T total return; cash earns the 1-month T-bill; 10 bp one-way on every unit of exposure change.
Out of sample from the leveraged fund's inception (nothing was tuned on any of these funds; the
penalties were sized on the US market factor).

Usage:
    python research/letf_trend_jump.py --ff <F-F daily CSV> --etf <1x adjclose.csv> --letf <leveraged adjclose.csv> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paris  # noqa: E402

from walkforward_jump import COST, load_ff, metrics, net_returns  # noqa: E402

PAIRS = {"SPY": ("UPRO", 3), "QQQ": ("TQQQ", 3), "XLK": ("TECL", 3), "IWM": ("TNA", 3),
         "EEM": ("EDC", 3), "TLT": ("TMF", 3), "GLD": ("UGL", 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff", required=True)
    ap.add_argument("--etf", required=True)
    ap.add_argument("--letf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rf = load_ff(args.ff)["RF"]
    px1 = pd.read_csv(args.etf, index_col=0, parse_dates=True)
    pxl = pd.read_csv(args.letf, index_col=0, parse_dates=True)
    md = [f"# Trend / gate / graded on the 1x ETF, invested in the leveraged fund — {pd.Timestamp.today().date()}\n",
          f"Signals on the 1x fund (trend λ=5, risk λ=50; rolling 1,260-day, monthly refit, lag 1). Exposure applied to the "
          f"leveraged fund's daily total return; T-bill when out; {COST * 1e4:.0f} bp one-way on exposure changes. "
          "Out of sample from the leveraged fund's inception. UGL is 2x (no 3x gold fund exists since 2020).\n"]
    summary = {}
    for base, (lev, mult) in PAIRS.items():
        r1 = px1[base].dropna().pct_change().dropna()
        rl = pxl[lev].dropna().pct_change().dropna()
        trend = paris.trend_states(r1)
        risk = paris.risk_states(r1)
        idx = rl.index.intersection(trend.dropna().index)
        rl, rf_ = rl.loc[idx], rf.reindex(idx).fillna(0.0)
        r1w = r1.reindex(idx)
        start = idx[0]
        rules = {
            "trend": trend.loc[idx],
            "gate": paris.combine_states(risk, trend, "gate").loc[idx],
            "graded": paris.combine_states(risk, trend, "graded").loc[idx],
        }
        rows = {}
        one = pd.Series(1.0, index=idx)
        rows[f"B&H {lev} ({mult}x)"] = metrics(net_returns(one, rl, rf_), one, rf_, start)
        rows[f"B&H {base} (1x)"] = metrics(net_returns(one, r1w, rf_), one, rf_, start)
        for name, e in rules.items():
            e = e.fillna(0.0)
            rows[f"{name} → {lev}"] = metrics(net_returns(e, rl, rf_), e, rf_, start)
        rows[f"trend → {base} (1x, reference)"] = metrics(net_returns(rules["trend"].fillna(0.0), r1w, rf_), rules["trend"].fillna(0.0), rf_, start)
        table = pd.DataFrame(rows).T
        table.index.name = f"{base}/{lev} {idx[0].date()}..{idx[-1].date()}"
        table.to_csv(out / f"{base}_{lev}.csv")
        md.append(f"\n## {base} → {lev} ({mult}x), {idx[0].date()} .. {idx[-1].date()}\n\n```\n{table.round(3).to_string()}\n```\n")
        print(table.round(3).to_string(), "\n", flush=True)
        summary[f"{base}/{lev}"] = {k: v for k, v in rows.items()}
    # cross-fund view per metric
    for metric in ("Sharpe", "MaxDD", "Martin", "CAGR", "Turnover/yr"):
        m = pd.DataFrame({pair: {("B&H lev" if k.startswith("B&H") and "1x" not in k else "B&H 1x" if k.startswith("B&H") else k.split(" →")[0] + (" 1x" if "reference" in k else " → lev")): v[metric]
                                 for k, v in rows_.items()} for pair, rows_ in summary.items()})
        m["mean"] = m.mean(axis=1)
        md.append(f"\n## {metric} across funds\n\n```\n{m.round(3 if metric != 'Turnover/yr' else 2).to_string()}\n```\n")
        print(f"=== {metric} ===\n{m.round(3).to_string()}\n", flush=True)
    (out / "LETF_TREND.md").write_text("\n".join(md))
    print("written", out / "LETF_TREND.md")


if __name__ == "__main__":
    main()
