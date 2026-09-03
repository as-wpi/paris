"""Four-state momentum cycle as a long/flat timing rule vs the jump-model trend switch.
Pre-registration: research/momentum_cycle_timing/preregistration.json.

    ~/.venvs/rtl-workspace/bin/python research/momentum_cycle_timing.py
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
from frozen_rule_benchmark import TICKERS, evaluate, folds, load_ohlc, load_rf  # noqa: E402

OUT = _HERE / "momentum_cycle_timing"
WARM_YEARS = 5


def exposure_from_states(states: pd.Series, a_co: pd.Series | float, a_re: pd.Series | float) -> pd.Series:
    """Long-only mapping e = (w+1)/2 of the paper's position; speeds may vary by date."""
    e = pd.Series(np.nan, index=states.index)
    e[states == "Bull"] = 1.0
    e[states == "Bear"] = 0.0
    co, re_ = states == "Correction", states == "Rebound"
    a_co = pd.Series(a_co, index=states.index) if np.isscalar(a_co) else a_co
    a_re = pd.Series(a_re, index=states.index) if np.isscalar(a_re) else a_re
    e[co] = 1.0 - a_co[co]
    e[re_] = a_re[re_]
    return e.shift(1)  # state at close T-1 -> exposure for T


def dyn_speeds_causal(r: pd.Series, states: pd.Series) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    first = states.first_valid_index()
    years = sorted(set(states.loc[first:].index.year))
    a_co = pd.Series(np.nan, index=states.index)
    a_re = pd.Series(np.nan, index=states.index)
    log = []
    for y in years[WARM_YEARS:]:
        upto = pd.Timestamp(f"{y - 1}-12-31")
        sp = paris.dynamic_speeds(r.loc[:upto])
        co = float(sp["Correction"]) if np.isfinite(sp["Correction"]) else 0.5
        re_ = float(sp["Rebound"]) if np.isfinite(sp["Rebound"]) else 0.5
        m = states.index.year == y
        a_co[m], a_re[m] = co, re_
        log.append({"year": y, "a_Co": round(co, 3), "a_Re": round(re_, 3)})
    return a_co, a_re, pd.DataFrame(log)


def arms_for(r: pd.Series) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    st = paris.momentum_states(r)
    fast = paris.momentum_signal(r, "fast")
    slow = paris.momentum_signal(r, "slow")
    a_co, a_re, speeds_log = dyn_speeds_causal(r, st)
    dyn = exposure_from_states(st, a_co, a_re).where(a_co.notna())
    arms = {
        "Buy & hold": pd.Series(1.0, index=r.index),
        "trend λ5 rolling": paris.trend_states(r, jump_penalty=5.0),
        "DYN speeds (causal)": dyn,
        "MED a=0.5": exposure_from_states(st, 0.5, 0.5),
        "Bull+Rebound": (fast >= 0).astype(float).where(fast.notna()).shift(1),
        "OutOnlyBear": (~st.isin(["Bear"])).astype(float).where(st.notna()).shift(1),
        "SlowSign": (slow >= 0).astype(float).where(slow.notna()).shift(1),
    }
    return arms, speeds_log


def paired(full, a, b, col, rng):
    pv = full.pivot(index="ticker", columns="arm", values=col)
    d = (pv[a] - pv[b]).dropna()
    boot = [np.median(rng.choice(d.values, len(d))) for _ in range(1000)]
    return float(d.median()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int((d > 0).sum())


def main() -> None:
    rf = load_rf()
    res, tables, logs = [], [], []
    for t in TICKERS:
        r = load_ohlc(t)["Close"].pct_change().dropna()
        arms, sl = arms_for(r)
        sl.insert(0, "ticker", t)
        logs.append(sl)
        res.append(evaluate(t, arms, r, rf))
        tb = paris.momentum_state_table(r, rf=rf.reindex(r.index).ffill().fillna(0.0))
        tb.insert(0, "ticker", t)
        tables.append(tb)
        print(t, "done", flush=True)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(OUT / "all_metrics.csv", index=False)
    pd.concat(logs).to_csv(OUT / "dynamic_speeds_by_year.csv", index=False)
    tabs = pd.concat(tables)
    tabs.to_csv(OUT / "state_tables.csv")
    fl = folds()
    rng = np.random.default_rng(0)
    full = res[res.fold == "FULL"]
    order = ["Buy & hold", "trend λ5 rolling", "DYN speeds (causal)", "MED a=0.5", "Bull+Rebound", "OutOnlyBear", "SlowSign"]
    L = [f"# Four-state momentum cycle as a timing rule — {pd.Timestamp.today().date()}\n",
         "Pre-registration: `preregistration.json`. Long-only mapping e = (w+1)/2; state at close T−1 sets day T; 10 bp one-way; T-bill cash.\n"]
    med = res.groupby(["fold", "arm"])[["Sharpe", "MaxDD", "Martin", "CAGR", "TimeIn", "Flips/yr"]].median()
    for fold in ["FULL"] + list(fl):
        tb = med.loc[fold].reindex(order)
        L.append(f"\n## {fold} (cross-ticker medians)\n\n| Arm | Sharpe | MaxDD | Martin | CAGR | Time in | Flips/yr |\n|---|---:|---:|---:|---:|---:|---:|")
        for arm, row in tb.iterrows():
            L.append(f"| {arm} | {row.Sharpe:.2f} | {row.MaxDD:.1%} | {row.Martin:.2f} | {row.CAGR:.1%} | {row.TimeIn:.2f} | {row['Flips/yr']:.1f} |")
    L.append("\n## Per-ticker full-OOS Sharpe\n\n| Ticker | " + " | ".join(order) + " |\n|---|" + "---:|" * len(order))
    pv = full.pivot(index="ticker", columns="arm", values="Sharpe")
    for t in TICKERS:
        L.append(f"| {t} | " + " | ".join(f"{pv.loc[t, a]:.2f}" for a in order) + " |")
    L.append("\n## Pre-registered decision rule (vs trend λ5 rolling)\n")
    verdict = {}
    for a in order[2:]:
        dm, lo, hi, wm = paired(full, a, "trend λ5 rolling", "Martin", rng)
        ds, slo, shi, ws = paired(full, a, "trend λ5 rolling", "Sharpe", rng)
        dbh, _, _, wbh = paired(full, a, "Buy & hold", "Sharpe", rng)
        bleed_ok, notes = True, []
        for fk, (s0, s1, kind) in fl.items():
            sub = res[res.fold == fk]
            if sub.empty:
                continue
            m = sub.groupby("arm")[["Sharpe", "TimeIn", "MaxDD"]].median()
            if kind == "bull":
                bl_a = m.loc["Buy & hold", "Sharpe"] - m.loc[a, "Sharpe"]
                bl_t = m.loc["Buy & hold", "Sharpe"] - m.loc["trend λ5 rolling", "Sharpe"]
                ok = bl_a <= bl_t + 0.10
                bleed_ok &= bool(ok)
                notes.append(f"{fk}: bleed {bl_a:+.2f} vs trend {bl_t:+.2f} {'ok' if ok else 'FAIL'}")
            else:
                notes.append(f"{fk}: MaxDD {m.loc[a,'MaxDD']:.1%} vs trend {m.loc['trend λ5 rolling','MaxDD']:.1%} (time-in {m.loc[a,'TimeIn']:.2f} vs {m.loc['trend λ5 rolling','TimeIn']:.2f})")
        promote = ((dm > 0 and lo > 0) or (wm >= 6 and ds >= 0)) and bleed_ok
        verdict[a] = "PROMOTE over trend" if promote else "RETIRE"
        L.append(f"\n**{a}: {verdict[a]}** — ΔMartin {dm:+.2f} (CI [{lo:+.2f}, {hi:+.2f}]), wins {wm}/8; ΔSharpe {ds:+.2f} (CI [{slo:+.2f}, {shi:+.2f}]), wins {ws}/8; vs B&H ΔSharpe {dbh:+.2f}, wins {wbh}/8; bleed {'pass' if bleed_ok else 'FAIL'}.")
        for n in notes:
            L.append(f"- {n}")
    L.append("\n## Dynamic speeds estimated each year (causal), median across funds\n")
    sl = pd.concat(logs)
    g = sl.groupby("year")[["a_Co", "a_Re"]].median().round(2)
    L.append("| Year | a_Co | a_Re |\n|---|---:|---:|")
    for y, row in g.iterrows():
        L.append(f"| {y} | {row.a_Co:.2f} | {row.a_Re:.2f} |")
    L.append("\n## Four-state conditional tables (per fund; full history; own annualised mean by state)\n")
    L.append("| Ticker | " + " | ".join(tabs.index.unique().astype(str)) + " |\n|---|" + "---:|" * len(tabs.index.unique()))
    col = [c for c in tabs.columns if "mean" in c.lower()][0]
    for t in TICKERS:
        sub = tabs[tabs.ticker == t]
        L.append(f"| {t} | " + " | ".join(f"{sub.loc[s, col]:.1%}" if s in sub.index else "" for s in tabs.index.unique()) + " |")
    L.append(f"\n(column shown: `{col}`; full tables in `state_tables.csv`)")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
