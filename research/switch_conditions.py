"""External conditions on the trend switch: VR(5,63) MA and the VIX term structure.
Pre-registration: research/switch_conditions/preregistration.json.

    ~/.venvs/rtl-workspace/bin/python research/switch_conditions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
import paris  # noqa: E402
from frozen_rule_benchmark import DATA, TICKERS, evaluate, folds, load_ohlc, load_rf  # noqa: E402

OUT = _HERE / "switch_conditions"
REF = "trend λ5 rolling"


def variance_ratio_ma(close: pd.Series, q: int = 5, window: int = 63, ma: int = 200) -> pd.Series:
    """Screener definition: Var(q-day log return) / (q · Var(1-day)) over ``window`` bars, then an
    ``ma``-bar simple moving average. Both variances are sample variances over the same window."""
    x = np.log(close).diff()
    xq = np.log(close).diff(q)
    vr = xq.rolling(window).var() / (q * x.rolling(window).var())
    return vr.rolling(ma).mean()


def vix_term() -> pd.Series:
    vix = pd.read_csv(DATA / "VIX.csv", index_col=0, parse_dates=True)["close"]
    v3 = pd.read_csv(DATA / "VIX3M.csv", index_col=0, parse_dates=True)["close"]
    return (v3 - vix).dropna()


def cond_at_t(series: pd.Series, idx: pd.DatetimeIndex, rule) -> pd.Series:
    """Condition known at close T-1, aligned to the fund's calendar (causal ffill), for day T."""
    c = rule(series).astype(float).where(series.notna())
    return c.shift(1).reindex(idx, method="ffill")


def main() -> None:
    rf = load_rf()
    spy_close = load_ohlc("SPY")["Close"]
    vr_spy = variance_ratio_ma(spy_close)
    term = vix_term()
    res = []
    for t in TICKERS:
        px = load_ohlc(t)
        r = px["Close"].pct_change().dropna()
        idx = r.index
        trend = paris.trend_states(r, jump_penalty=5.0)
        c_vr_own = cond_at_t(variance_ratio_ma(px["Close"]), idx, lambda s: s >= 1.0)
        c_vr_spy = cond_at_t(vr_spy, idx, lambda s: s >= 1.0)
        c_vix = cond_at_t(term, idx, lambda s: s >= 0.0)
        arms = {
            "Buy & hold": pd.Series(1.0, index=idx), REF: trend,
            "trend x VR own": trend * c_vr_own, "trend x VR SPY": trend * c_vr_spy, "trend x VIX term": trend * c_vix,
            "VR own alone": c_vr_own, "VR SPY alone": c_vr_spy, "VIX term alone": c_vix,
        }
        # VIX arms: NaN before 2006-07 -> evaluate() starts OOS at the first date every arm is valid,
        # which would shorten the non-VIX arms too; so evaluate the VIX arms separately and merge.
        base = {k: v for k, v in arms.items() if "VIX" not in k}
        vixa = {k: v for k, v in arms.items() if "VIX" in k or k in ("Buy & hold", REF)}
        e1 = evaluate(t, base, r, rf)
        e1["panel"] = "VR"
        e2 = evaluate(t, vixa, r, rf)
        e2["panel"] = "VIX"
        res += [e1, e2]
        print(t, "done", flush=True)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(OUT / "all_metrics.csv", index=False)
    fl = folds()
    rng = np.random.default_rng(0)
    L = [f"# External conditions on the trend switch — {pd.Timestamp.today().date()}\n",
         f"Pre-registration: `preregistration.json`. Reference switch: {REF}. Conditions at close T−1; 10 bp; T-bill cash.\n"]
    verdict = {}
    for panel, order in (("VR", ["Buy & hold", REF, "trend x VR own", "trend x VR SPY", "VR own alone", "VR SPY alone"]),
                         ("VIX", ["Buy & hold", REF, "trend x VIX term", "VIX term alone"])):
        sub = res[res.panel == panel]
        full = sub[sub.fold == "FULL"]
        start = full.groupby("ticker")["start"].first()
        L.append(f"\n# Panel {panel} (OOS starts: " + ", ".join(f"{t} {s}" for t, s in start.items()) + ")\n")
        med = sub.groupby(["fold", "arm"])[["Sharpe", "MaxDD", "Martin", "CAGR", "TimeIn", "Flips/yr"]].median()
        for fold in ["FULL"] + list(fl):
            if fold not in med.index.get_level_values(0):
                continue
            tb = med.loc[fold].reindex(order).dropna(how="all")
            L.append(f"\n## {fold} (cross-ticker medians)\n\n| Arm | Sharpe | MaxDD | Martin | CAGR | Time in | Flips/yr |\n|---|---:|---:|---:|---:|---:|---:|")
            for arm, row in tb.iterrows():
                L.append(f"| {arm} | {row.Sharpe:.2f} | {row.MaxDD:.1%} | {row.Martin:.2f} | {row.CAGR:.1%} | {row.TimeIn:.2f} | {row['Flips/yr']:.1f} |")
        pv = {c: full.pivot(index="ticker", columns="arm", values=c) for c in ("Martin", "Sharpe", "MaxDD")}
        L.append(f"\n## Decision rule, panel {panel} (vs {REF})\n")
        for a in [x for x in order if x.startswith("trend x")]:
            out = {}
            for c in ("Martin", "Sharpe", "MaxDD"):
                d = (pv[c][a] - pv[c][REF]).dropna()
                boot = [np.median(rng.choice(d.values, len(d))) for _ in range(1000)]
                out[c] = (float(d.median()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int((d > 0).sum()))
            bleed_ok, notes = True, []
            for fk, (s0, s1, kind) in fl.items():
                sf = sub[sub.fold == fk]
                if sf.empty:
                    continue
                m = sf.groupby("arm")[["Sharpe", "TimeIn", "MaxDD"]].median()
                if a not in m.index:
                    continue
                if kind == "bull":
                    bl_a = m.loc["Buy & hold", "Sharpe"] - m.loc[a, "Sharpe"]
                    bl_t = m.loc["Buy & hold", "Sharpe"] - m.loc[REF, "Sharpe"]
                    ok = bl_a <= bl_t + 0.10
                    bleed_ok &= bool(ok)
                    notes.append(f"{fk}: bleed {bl_a:+.2f} vs reference {bl_t:+.2f} {'ok' if ok else 'FAIL'}")
                else:
                    notes.append(f"{fk}: MaxDD {m.loc[a,'MaxDD']:.1%} vs reference {m.loc[REF,'MaxDD']:.1%} (time-in {m.loc[a,'TimeIn']:.2f} vs {m.loc[REF,'TimeIn']:.2f})")
            dm, lo, hi, wm = out["Martin"]
            ds, slo, shi, ws = out["Sharpe"]
            promote = ((dm > 0 and lo > 0) or (wm >= 6 and ds >= 0)) and bleed_ok
            verdict[a] = "PROMOTE" if promote else "RETIRE"
            L.append(f"\n**{a}: {verdict[a]}** — ΔMartin {dm:+.2f} (CI [{lo:+.2f}, {hi:+.2f}]), wins {wm}/8; ΔSharpe {ds:+.2f} (CI [{slo:+.2f}, {shi:+.2f}]), wins {ws}/8; ΔMaxDD {out['MaxDD'][0]:+.1%}; bleed {'pass' if bleed_ok else 'FAIL'}.")
            for n in notes:
                L.append(f"- {n}")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
