"""Calibration memory: rolling 1,260-day (incumbent) vs expanding-window calibration of the PARIS
jump models. Pre-registration: research/calibration_memory/preregistration.json.

    ~/.venvs/rtl-workspace/bin/python research/calibration_memory.py
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
from frozen_rule_benchmark import EXUS, TICKERS, evaluate, folds, load_ohlc, load_rf  # noqa: E402

OUT = _HERE / "calibration_memory"
PREREG = json.loads((OUT / "preregistration.json").read_text())


def arms_for(r: pd.Series) -> dict[str, pd.Series]:
    return {
        "Buy & hold": pd.Series(1.0, index=r.index),
        "trend rolling 1260": paris.trend_states(r, jump_penalty=5.0),
        "trend expanding": paris.trend_states(r, jump_penalty=5.0, calibration="expanding"),
        "risk rolling 1260": paris.risk_states(r, jump_penalty=50.0),
        "risk expanding": paris.risk_states(r, jump_penalty=50.0, calibration="expanding"),
    }


def paired(full: pd.DataFrame, a: str, b: str, col: str, rng) -> tuple[float, float, float, int]:
    d = (full.pivot(index="ticker", columns="arm", values=col)[a] - full.pivot(index="ticker", columns="arm", values=col)[b]).dropna()
    boot = [np.median(rng.choice(d.values, len(d))) for _ in range(1000)]
    return float(d.median()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int((d > 0).sum())


def main() -> None:
    rf = load_rf()
    res, expos = [], {}
    for t in TICKERS:
        r = load_ohlc(t)["Close"].pct_change().dropna()
        arms = arms_for(r)
        expos[t] = pd.DataFrame(arms)
        res.append(evaluate(t, arms, r, rf))
        print(t, "done", flush=True)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(OUT / "all_metrics.csv", index=False)
    fl = folds()
    rng = np.random.default_rng(0)
    full = res[res.fold == "FULL"]
    L = [f"# Calibration memory: rolling vs expanding — {pd.Timestamp.today().date()}\n",
         "Pre-registration: `preregistration.json`. Same universe, folds and execution as the frozen-rule benchmark.\n"]
    med = res.groupby(["fold", "arm"])[["Sharpe", "MaxDD", "Martin", "CAGR", "TimeIn", "Flips/yr"]].median()
    order = ["Buy & hold", "trend rolling 1260", "trend expanding", "risk rolling 1260", "risk expanding"]
    for fold in ["FULL"] + list(fl):
        tb = med.loc[fold].reindex(order)
        L.append(f"\n## {fold} (cross-ticker medians)\n\n| Arm | Sharpe | MaxDD | Martin | CAGR | Time in | Flips/yr |\n|---|---:|---:|---:|---:|---:|---:|")
        for arm, row in tb.iterrows():
            L.append(f"| {arm} | {row.Sharpe:.2f} | {row.MaxDD:.1%} | {row.Martin:.2f} | {row.CAGR:.1%} | {row.TimeIn:.2f} | {row['Flips/yr']:.1f} |")
    L.append("\n## Per-ticker full-OOS (Sharpe / Martin / MaxDD)\n\n| Ticker | " + " | ".join(order[1:]) + " |\n|---|" + "---:|" * 4)
    for t in TICKERS:
        sub = full[full.ticker == t].set_index("arm")
        L.append(f"| {t} | " + " | ".join(f"{sub.loc[a,'Sharpe']:.2f} / {sub.loc[a,'Martin']:.2f} / {sub.loc[a,'MaxDD']:.0%}" for a in order[1:]) + " |")
    # decision rule
    L.append("\n## Pre-registered decision rule\n")
    verdict = {}
    for model in ("trend", "risk"):
        a, b = f"{model} expanding", f"{model} rolling 1260"
        dm, lo, hi, wins = paired(full, a, b, "Martin", rng)
        ds, slo, shi, swins = paired(full, a, b, "Sharpe", rng)
        dd, _, _, dwins = paired(full, a, b, "MaxDD", rng)
        bleed_ok, notes = True, []
        bh = res[res.arm == "Buy & hold"]
        for fk, (s0, s1, kind) in fl.items():
            sub = res[res.fold == fk]
            if sub.empty:
                continue
            m = sub.groupby("arm")[["Sharpe", "TimeIn", "MaxDD"]].median()
            if kind == "bull":
                bl_a = m.loc["Buy & hold", "Sharpe"] - m.loc[a, "Sharpe"]
                bl_b = m.loc["Buy & hold", "Sharpe"] - m.loc[b, "Sharpe"]
                ok = bl_a <= bl_b + 0.10
                bleed_ok &= bool(ok)
                notes.append(f"{fk}: bleed vs B&H {bl_a:+.2f} (expanding) vs {bl_b:+.2f} (rolling) {'ok' if ok else 'FAIL'}; time-in {m.loc[a,'TimeIn']:.2f} vs {m.loc[b,'TimeIn']:.2f}")
            else:
                if m.loc[a, "TimeIn"] >= 0.5 and m.loc[b, "TimeIn"] >= 0.5:
                    notes.append(f"{fk}: MaxDD {m.loc[a,'MaxDD']:.1%} vs {m.loc[b,'MaxDD']:.1%}; Sharpe {m.loc[a,'Sharpe']:.2f} vs {m.loc[b,'Sharpe']:.2f}")
                else:
                    notes.append(f"{fk}: both mostly flat (time-in {m.loc[a,'TimeIn']:.2f} vs {m.loc[b,'TimeIn']:.2f}); MaxDD {m.loc[a,'MaxDD']:.1%} vs {m.loc[b,'MaxDD']:.1%} reported only")
        promote = ((dm > 0 and lo > 0) or wins >= 6) and ds >= 0 and bleed_ok
        v = "PROMOTE to default candidate" if promote else "RETIRE"
        if model == "risk":
            v += " (diagnostic model under D1; not a promotion)"
        verdict[model] = v
        L.append(f"\n**{a} vs {b}: {v}** — ΔMartin {dm:+.2f} (CI [{lo:+.2f}, {hi:+.2f}]), wins {wins}/8; ΔSharpe {ds:+.2f} (CI [{slo:+.2f}, {shi:+.2f}]), wins {swins}/8; ΔMaxDD {dd:+.1%} (better {dwins}/8); bleed condition {'pass' if bleed_ok else 'FAIL'}.")
        for n in notes:
            L.append(f"- {n}")
    # named diagnostics
    L.append("\n## Named diagnostics\n")
    spy = expos["SPY"]
    for a in ("risk rolling 1260", "risk expanding"):
        L.append(f"- SPY {a}: time-in 2019–2022 = {spy[a].loc['2019':'2022'].mean():.2f}; by year " +
                 ", ".join(f"{y}: {v:.2f}" for y, v in spy[a].loc['2019':'2022'].groupby(spy[a].loc['2019':'2022'].index.year).mean().items()))
    for a in ("trend rolling 1260", "trend expanding"):
        off = np.median([1 - expos[t][a].loc['2023':'2024'].mean() for t in TICKERS])
        L.append(f"- {a}: median off-fraction 2023–24 = {off:.2f}")
    tlt = expos["TLT"]
    for a in ("risk rolling 1260", "risk expanding", "trend rolling 1260", "trend expanding"):
        L.append(f"- TLT {a}: time-in 2022–2026 = {tlt[a].loc['2022':].mean():.2f}")
    L.append("\n## Attribution: US vs ex-US (full-OOS median ΔSharpe, expanding − rolling)\n\n| Model | US | ex-US |\n|---|---:|---:|")
    for model in ("trend", "risk"):
        pv = full.pivot(index="ticker", columns="arm", values="Sharpe")
        d = pv[f"{model} expanding"] - pv[f"{model} rolling 1260"]
        L.append(f"| {model} | {d.reindex([t for t in TICKERS if t not in EXUS]).median():+.2f} | {d.reindex(EXUS).median():+.2f} |")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    for t, e in expos.items():
        e.to_csv(OUT / f"{t}_exposures.csv")
    print("\n".join(L))


if __name__ == "__main__":
    main()
