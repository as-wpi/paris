"""State-dependent calibration memory (hybrid): expanding scaler + on-centre, rolling off-centre.
Pre-registration: research/hybrid_memory/preregistration.json.

    ~/.venvs/rtl-workspace/bin/python research/hybrid_memory.py
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
from paris.jump import _loss, _online_last, _rolling_fits, _standardise  # noqa: E402
from frozen_rule_benchmark import EXUS, TICKERS, evaluate, folds, load_ohlc, load_rf  # noqa: E402

OUT = _HERE / "hybrid_memory"
WINDOW = 1260


def hybrid_states(features: pd.DataFrame, lam: float, off_state: int) -> pd.Series:
    """Two-state online labels with the expanding scaler and on-centre and the rolling off-centre."""
    df, X, w, lookback, fits_r = _rolling_fits(features, lam, WINDOW, "ME", 2, None, 10, 0, 3.0, None, "rolling")
    _, _, _, _, fits_e = _rolling_fits(features, lam, WINDOW, "ME", 2, None, 10, 0, 3.0, None, "expanding")
    assert [f[0] for f in fits_r] == [f[0] for f in fits_e]
    n = len(df)
    out = np.full(n, np.nan)
    on_state = 1 - off_state
    for j, ((start, mu_r, sd_r, c_r), (_, mu_e, sd_e, c_e)) in enumerate(zip(fits_r, fits_e)):
        off_orig = c_r[off_state] * sd_r + mu_r              # rolling off-centre, original units
        centers = c_e.copy()
        centers[off_state] = (off_orig - mu_e) / sd_e         # in the expanding scaler's units
        centers[on_state] = c_e[on_state]
        stop = fits_r[j + 1][0] if j + 1 < len(fits_r) else n
        for t in range(start, stop):
            lo = max(0, t - lookback + 1)
            Z = _standardise(X[lo:t + 1], mu_e, sd_e, 3.0)
            out[t] = _online_last(_loss(Z, centers), lam)
    return pd.Series(out, index=df.index)


def arms_for(r: pd.Series) -> dict[str, pd.Series]:
    slow, fast = paris.trend_signal(r, "slow"), paris.trend_signal(r, "fast")
    tf = pd.DataFrame({"slow": slow, "fast": fast}).dropna()
    lv = paris.risk_signal(r, log=True).dropna().to_frame("logvol")
    th = hybrid_states(tf, 5.0, off_state=0).reindex(r.index).shift(1)          # state 1 = on
    rh = (1.0 - hybrid_states(lv, 50.0, off_state=1)).reindex(r.index).shift(1)  # state 0 = on
    return {
        "Buy & hold": pd.Series(1.0, index=r.index),
        "trend rolling 1260": paris.trend_states(r, jump_penalty=5.0),
        "trend expanding": paris.trend_states(r, jump_penalty=5.0, calibration="expanding"),
        "trend hybrid": th,
        "risk rolling 1260": paris.risk_states(r, jump_penalty=50.0),
        "risk expanding": paris.risk_states(r, jump_penalty=50.0, calibration="expanding"),
        "risk hybrid": rh,
    }


def paired(full, a, b, col, rng):
    pv = full.pivot(index="ticker", columns="arm", values=col)
    d = (pv[a] - pv[b]).dropna()
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
    order = ["Buy & hold", "trend rolling 1260", "trend expanding", "trend hybrid", "risk rolling 1260", "risk expanding", "risk hybrid"]
    L = [f"# Hybrid calibration memory — {pd.Timestamp.today().date()}\n", "Pre-registration: `preregistration.json`.\n"]
    med = res.groupby(["fold", "arm"])[["Sharpe", "MaxDD", "Martin", "CAGR", "TimeIn", "Flips/yr"]].median()
    for fold in ["FULL"] + list(fl):
        tb = med.loc[fold].reindex(order)
        L.append(f"\n## {fold} (cross-ticker medians)\n\n| Arm | Sharpe | MaxDD | Martin | CAGR | Time in | Flips/yr |\n|---|---:|---:|---:|---:|---:|---:|")
        for arm, row in tb.iterrows():
            L.append(f"| {arm} | {row.Sharpe:.2f} | {row.MaxDD:.1%} | {row.Martin:.2f} | {row.CAGR:.1%} | {row.TimeIn:.2f} | {row['Flips/yr']:.1f} |")
    L.append("\n## Pre-registered decision rule (hybrid vs rolling)\n")
    verdict = {}
    for model in ("trend", "risk"):
        a, b = f"{model} hybrid", f"{model} rolling 1260"
        dm, lo, hi, wm = paired(full, a, b, "Martin", rng)
        ds, slo, shi, ws = paired(full, a, b, "Sharpe", rng)
        dd, _, _, wd = paired(full, a, b, "MaxDD", rng)
        bleed_ok, notes = True, []
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
                notes.append(f"{fk}: bleed {bl_a:+.2f} vs rolling {bl_b:+.2f} {'ok' if ok else 'FAIL'}; time-in {m.loc[a,'TimeIn']:.2f} vs {m.loc[b,'TimeIn']:.2f}")
            else:
                notes.append(f"{fk}: MaxDD {m.loc[a,'MaxDD']:.1%} vs rolling {m.loc[b,'MaxDD']:.1%} vs expanding {m.loc[f'{model} expanding','MaxDD']:.1%} (time-in {m.loc[a,'TimeIn']:.2f} / {m.loc[b,'TimeIn']:.2f} / {m.loc[f'{model} expanding','TimeIn']:.2f})")
        promote = ((dm > 0 and lo > 0) or wm >= 6) and ds >= 0 and bleed_ok
        verdict[model] = ("PROMOTE" if promote else "RETIRE") + (" (diagnostic under D1)" if model == "risk" else "")
        L.append(f"\n**{a} vs {b}: {verdict[model]}** — ΔMartin {dm:+.2f} (CI [{lo:+.2f}, {hi:+.2f}]), wins {wm}/8; ΔSharpe {ds:+.2f} (CI [{slo:+.2f}, {shi:+.2f}]), wins {ws}/8; ΔMaxDD {dd:+.1%} (better {wd}/8); bleed {'pass' if bleed_ok else 'FAIL'}.")
        for n in notes:
            L.append(f"- {n}")
    L.append("\n## Named diagnostics\n")
    spy = expos["SPY"]
    for a in ("risk rolling 1260", "risk expanding", "risk hybrid"):
        L.append(f"- SPY {a}: time-in 2019–2022 = {spy[a].loc['2019':'2022'].mean():.2f}")
    for a in ("trend rolling 1260", "trend expanding", "trend hybrid"):
        off = np.median([1 - expos[t][a].loc['2023':'2024'].mean() for t in TICKERS])
        L.append(f"- {a}: median off-fraction 2023–24 = {off:.2f}")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    for t, e in expos.items():
        e.to_csv(OUT / f"{t}_exposures.csv")
    print("\n".join(L))


if __name__ == "__main__":
    main()
