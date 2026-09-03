"""Trend-on/off as the on/off gate on top of the vol-target sizing of a leveraged ETF.

Replicates the construction of the leveraged-ETF timing study §14.1 (UPRO_TQQQ_TIMING_STUDY.md):
w* = clip(sigma_target / sigma_LETF, 0, 1) with sigma_target = 30 % and sigma_LETF the RiskMetrics
EWMA (lambda 0.94, 60-day warm-up) of the LEVERAGED fund's own daily log returns; hysteresis band on
implied portfolio vol w * sigma in [25 %, 35 %] (no trade inside the band, weight drifts with
returns); rebalance only on a gate flip or a band exit; t+0 execution; 5 bp cost + 3 bp slippage
one-way on every unit traded. Gates compared, all evaluated on the 1x fund at the close of T-1:
none (pure vol-target), the study's 200-day SMA, and the jump-model trend-on/off (penalty 5).
Cash earns the 1-month T-bill (the study used 0 % / BIL). Out of sample from the leveraged fund's
inception; nothing was tuned on these funds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paris  # noqa: E402

from walkforward_jump import load_ff, metrics  # noqa: E402

PAIRS = {"SPY": ("UPRO", 3), "QQQ": ("TQQQ", 3), "XLK": ("TECL", 3), "IWM": ("TNA", 3),
         "EEM": ("EDC", 3), "TLT": ("TMF", 3), "GLD": ("UGL", 2)}
SIGMA_TARGET, BAND = 0.30, (0.25, 0.35)
COST_ONE_WAY = (5.0 + 3.0) / 1e4


def ewma_vol(r: pd.Series, lam: float = 0.94, warmup: int = 60) -> pd.Series:
    x = np.log1p(r)
    var = (x**2).ewm(alpha=1 - lam, adjust=False).mean()
    var.iloc[:warmup] = np.nan
    return np.sqrt(var * 252)


def run(gate: pd.Series, rl: pd.Series, rf: pd.Series, sigma: pd.Series, use_band: bool = True) -> tuple[pd.Series, pd.Series]:
    """Daily loop. ``gate``, ``sigma`` are known at the close of T-1 for day T (already lagged)."""
    n = len(rl)
    w = np.zeros(n)          # weight held during day t
    ret = np.zeros(n)
    w_prev = 0.0
    g, s, r, f = gate.to_numpy(), sigma.to_numpy(), rl.to_numpy(), rf.to_numpy()
    for t in range(n):
        target = w_prev
        if not (g[t] == 1) or np.isnan(s[t]):
            target = 0.0
        else:
            w_star = min(SIGMA_TARGET / s[t], 1.0) if s[t] > 0 else 0.0
            if w_prev == 0.0:
                target = w_star                                   # gate turned on: enter at w*
            elif use_band:
                implied = w_prev * s[t]
                if implied < BAND[0] or implied > BAND[1]:
                    target = w_star                               # band exit: rebalance
            else:
                target = w_star
        cost = abs(target - w_prev) * COST_ONE_WAY
        w[t] = target
        ret[t] = target * r[t] + (1 - target) * f[t] - cost
        # drift the held weight with the day's returns for tomorrow's band test
        gross = 1 + target * r[t] + (1 - target) * f[t]
        w_prev = target * (1 + r[t]) / gross if gross > 0 else 0.0
    return pd.Series(ret, index=rl.index), pd.Series(w, index=rl.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff", required=True)
    ap.add_argument("--etf", required=True)
    ap.add_argument("--letf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rf_all = load_ff(args.ff)["RF"]
    px1 = pd.read_csv(args.etf, index_col=0, parse_dates=True)
    pxl = pd.read_csv(args.letf, index_col=0, parse_dates=True)
    md = [f"# Trend gate on top of the vol-target sizing (study §14.1 construction) — {pd.Timestamp.today().date()}\n",
          "w* = clip(30 % / EWMA-vol of the leveraged fund, 0, 1); band [25 %, 35 %] on implied vol; rebalance on gate flip or "
          "band exit; t+0; 8 bp one-way; T-bill cash. Gates on the 1x fund at the close of T-1. OOS from the leveraged fund's inception.\n"]
    summary = {}
    for base, (lev, mult) in PAIRS.items():
        p1 = px1[base].dropna()
        r1 = p1.pct_change().dropna()
        rl = pxl[lev].dropna().pct_change().dropna()
        sigma = ewma_vol(rl).shift(1)                                    # LETF vol known at close T-1
        trend = paris.trend_states(r1)                                   # lag 1 built in
        sma_on = (p1 > p1.rolling(200).mean()).astype(float).shift(1)    # 200SMA on the 1x, known at T-1
        idx = rl.index.intersection(trend.dropna().index).intersection(sigma.dropna().index)
        rl_, rf_ = rl.loc[idx], rf_all.reindex(idx).fillna(0.0)
        start = idx[0]
        rows, weights = {}, {}
        one = pd.Series(1.0, index=idx)
        rows["B&H leveraged"] = metrics(rl_ - 0.0, one, rf_, start)
        for label, gate, band in (("VT30, no gate", one, True), ("200SMA gate + VT30 (study headline)", sma_on.reindex(idx).fillna(0.0), True),
                                  ("trend gate + VT30", trend.reindex(idx).fillna(0.0), True),
                                  ("trend gate + VT30, no band", trend.reindex(idx).fillna(0.0), False),
                                  ("trend gate, unsized (w=1)", trend.reindex(idx).fillna(0.0), None)):
            if band is None:
                e = gate
                ret = e * rl_ + (1 - e) * rf_ - e.diff().abs().fillna(0.0) * COST_ONE_WAY
                w = e
            else:
                ret, w = run(gate, rl_, rf_, sigma.reindex(idx), use_band=band)
            m = metrics(ret, w, rf_, start)
            m["Realised vol"] = m.pop("Vol")
            m["Avg weight"] = float(w.mean())
            rows[label] = m
            weights[label] = w
        table = pd.DataFrame(rows).T
        table.index.name = f"{base}/{lev} {idx[0].date()}..{idx[-1].date()}"
        table.to_csv(out / f"{base}_{lev}.csv")
        md.append(f"\n## {base} → {lev} ({mult}x), {idx[0].date()} .. {idx[-1].date()}\n\n```\n{table.round(3).to_string()}\n```\n")
        print(table.round(3).to_string(), "\n", flush=True)
        summary[f"{base}/{lev}"] = rows
    for metric in ("Sharpe", "MaxDD", "Martin", "CAGR", "Turnover/yr", "Avg weight"):
        m = pd.DataFrame({pair: {k: v.get(metric, np.nan) for k, v in rows_.items()} for pair, rows_ in summary.items()})
        m["mean"] = m.mean(axis=1)
        md.append(f"\n## {metric} across funds\n\n```\n{m.round(3).to_string()}\n```\n")
        print(f"=== {metric} ===\n{m.round(3).to_string()}\n", flush=True)
    (out / "LETF_VOLTARGET_TREND.md").write_text("\n".join(md))
    print("written", out / "LETF_VOLTARGET_TREND.md")


if __name__ == "__main__":
    main()
