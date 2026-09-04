"""Crash speed — test H (Grossman-Zhou drawdown governor on the stack's P&L).
Pre-registration: Strategy Research Memos/crash-speed-switch-vs-sizer/preregistration_H.json.

    ~/.venvs/rtl-workspace/bin/python research/crash_speed_H.py
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
from crash_speed_AB import COST, FAST, MEMO, PAIRS, ep_stats, episodes, ewma_vol, load, vt_path  # noqa: E402
from frozen_rule_benchmark import FF, TICKERS, load_rf  # noqa: E402
from walkforward_jump import load_ff  # noqa: E402

OUT = _HERE / "crash_speed_H"
CALM = ["K1995_99_bull", "K2013_14_bull", "A_bull_2021", "D_recent_2025plus"]
GZ = {"H_parity": (0.60, 1.25, 252), "H_full": (0.60, 2.5, 252)}


def governed(w_stack: pd.Series, r: pd.Series, rf: pd.Series, alpha: float, m: float, window: int):
    """Governor on the strategy's OWN wealth: g_t from wealth through T-1 scales the stack weight for T."""
    n = len(r)
    w = np.zeros(n); ret = np.zeros(n)
    W = 1.0; hist = [1.0]; w_prev = 0.0
    ws, rr, f = w_stack.to_numpy(), r.to_numpy(), rf.to_numpy()
    for t in range(n):
        peak = max(hist[-window:])
        g = min(max(m * (W - alpha * peak) / W, 0.0), 1.0) if W > 0 else 0.0
        tgt = ws[t] * g if np.isfinite(ws[t]) else 0.0
        cost = abs(tgt - w_prev) * COST
        ret[t] = tgt * rr[t] + (1 - tgt) * f[t] - cost
        w[t] = tgt
        W *= 1 + ret[t]
        hist.append(W)
        gross = 1 + tgt * rr[t] + (1 - tgt) * f[t]
        w_prev = tgt * (1 + rr[t]) / gross if gross > 0 else 0.0
    return pd.Series(w, index=r.index), pd.Series(ret, index=r.index)


def run_vehicle(name, sig_r, veh, rf, target, band):
    rv = veh["C"].pct_change().dropna(); idx = rv.index
    rf_ = rf.reindex(idx).ffill().fillna(0.0)
    trend0 = paris.trend_states(sig_r, jump_penalty=5.0, lag=0)
    g_inc = trend0.reindex(idx).ffill().shift(1).fillna(0.0)
    s94 = ewma_vol(rv).shift(1).reindex(idx)
    w_inc, ret_inc = vt_path(g_inc, s94, rv, rf_, target, band)
    start = max(trend0.reindex(idx).first_valid_index(), ret_inc.first_valid_index())
    rv2, rf2, w_inc2 = rv.loc[start:], rf_.loc[start:], w_inc.loc[start:]
    arms = {"incumbent": (w_inc2, ret_inc.loc[start:])}
    for k, (a, m, win) in GZ.items():
        arms[k] = governed(w_inc2, rv2, rf2, a, m, win)
    bh = rv2.copy(); bh.iloc[0] -= COST
    arms["buy_hold"] = (pd.Series(1.0, index=rv2.index), bh)
    rows = []
    for label, (w, ret) in arms.items():
        rows.append({"vehicle": name, "arm": label, "episode": "FULL", "Sharpe": float(paris.sharpe(ret, rf=rf2)), "CAGR": float(paris.cagr(ret)),
                     "Martin": float(paris.martin_ratio(ret)), "MaxDD": float(paris.max_drawdown(ret)), "MeanW": float(w.mean()),
                     "Turnover/yr": float(w.diff().abs().sum() / (len(w) / 252))})
        for ek, (a, b, grp) in episodes().items():
            if a < start:
                continue
            st = ep_stats(ret, w, a, b)
            if st:
                seg = ret.loc[a:b]; st["Sharpe"] = float(paris.sharpe(seg, rf=rf2.loc[seg.index]))
                rows.append({"vehicle": name, "arm": label, "episode": ek, "group": grp, **st})
    return pd.DataFrame(rows)


def report(res):
    rng = np.random.default_rng(0)
    full = res[res.episode == "FULL"]; ep = res[res.episode != "FULL"]
    L = [f"# Crash speed — test H (drawdown governor) — {pd.Timestamp.today().date()}\n", "Pre-registration: `preregistration_H.json`. Two separate verdicts: crash-speed (the sequence's rule) and governor.\n"]
    L.append("## Full sample (medians across 16 vehicles)\n\n| Arm | Sharpe | CAGR | Martin | MaxDD | Mean weight | Turnover/yr |\n|---|---:|---:|---:|---:|---:|---:|")
    for arm in ("incumbent", "H_parity", "H_full", "buy_hold"):
        m = full[full.arm == arm][["Sharpe", "CAGR", "Martin", "MaxDD", "MeanW", "Turnover/yr"]].median()
        L.append(f"| {arm} | {m.Sharpe:.2f} | {m.CAGR:.1%} | {m.Martin:.2f} | {m.MaxDD:.1%} | {m.MeanW:.2f} | {m['Turnover/yr']:.1f} |")
    L.append("\n## Episodes — median MaxDD by arm\n\n| Episode | incumbent | H_parity | H_full |\n|---|---:|---:|---:|")
    for e in FAST + ["E2007_09_gfc", "E2000_02_dotcom", "B_bear_2022", "E2011_eu_us"] + CALM:
        row = [ep[(ep.episode == e) & (ep.arm == a)]["MaxDD"].median() for a in ("incumbent", "H_parity", "H_full")]
        if all(np.isnan(row)):
            continue
        L.append(f"| {e} | " + " | ".join("—" if np.isnan(x) else f"{x:.1%}" for x in row) + " |")
    L.append("\n## Verdicts\n")
    verdict = {}
    for arm in ("H_parity", "H_full"):
        pvS = full.pivot(index="vehicle", columns="arm", values="Sharpe"); pvM = full.pivot(index="vehicle", columns="arm", values="Martin"); pvD = full.pivot(index="vehicle", columns="arm", values="MaxDD")
        ds = (pvS[arm] - pvS["incumbent"]).dropna(); dm = (pvM[arm] - pvM["incumbent"]).dropna(); ddf = (pvD[arm] - pvD["incumbent"]).dropna()
        bs = [np.median(rng.choice(ds.values, len(ds))) for _ in range(1000)]; bm = [np.median(rng.choice(dm.values, len(dm))) for _ in range(1000)]
        dd = []
        for e in FAST:
            a = ep[(ep.episode == e) & (ep.arm == arm)].set_index("vehicle")["MaxDD"]; i = ep[(ep.episode == e) & (ep.arm == "incumbent")].set_index("vehicle")["MaxDD"]
            x = (a.reindex(i.index) - i).dropna()
            if len(x):
                dd.append(float(x.median()))
        ddm = float(np.median(dd)) if dd else np.nan
        tr = (full.pivot(index="vehicle", columns="arm", values="Turnover/yr")[arm] / full.pivot(index="vehicle", columns="arm", values="Turnover/yr")["incumbent"]).median()
        bleed_ok, notes = True, []
        for e in CALM:
            sub = ep[ep.episode == e].pivot(index="vehicle", columns="arm", values="Sharpe")
            bl_a = (sub["buy_hold"] - sub[arm]).dropna().median(); bl_i = (sub["buy_hold"] - sub["incumbent"]).dropna().median()
            ok = bl_a <= bl_i + 0.10; bleed_ok &= bool(ok)
            notes.append(f"{e}: bleed vs B&H {bl_a:+.2f} (arm) vs {bl_i:+.2f} (incumbent) {'ok' if ok else 'FAIL'}")
        crash = (ddm >= 0.02) and (ds.median() >= -0.02) and bleed_ok and (tr <= 1.25)
        gov = (dm.median() > 0 and np.percentile(bm, 2.5) > 0) and (int((ddf > 0).sum()) >= 12) and bleed_ok
        verdict[arm] = {"crash_speed": "PASS" if crash else "RETIRE", "governor": "PASS" if gov else "RETIRE"}
        L.append(f"\n**{arm} — crash-speed: {verdict[arm]['crash_speed']}; governor: {verdict[arm]['governor']}** — fast-crash ΔMaxDD {ddm*100:+.1f} pts (bar +2.0); ΔSharpe {ds.median():+.3f} (CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]); ΔMartin {dm.median():+.2f} (CI [{np.percentile(bm,2.5):+.2f}, {np.percentile(bm,97.5):+.2f}]); full-sample MaxDD better in {int((ddf > 0).sum())}/{len(ddf)} (median {ddf.median()*100:+.1f} pts); turnover ×{tr:.2f}; bleed {'pass' if bleed_ok else 'FAIL'}.")
        for n_ in notes:
            L.append(f"- {n_}")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n"); (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n".join(L))


def main():
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        report(pd.read_csv(OUT / "all_results.csv")); return
    rf = load_rf(); ff = load_ff(str(FF))
    pan = json.loads((MEMO / "preregistration_AB.json").read_text())["stack_under_study"]["panels"]
    res = []
    mkt = pd.DataFrame({"C": (1 + ff["MKT"].loc["1926":]).cumprod()})
    r0 = run_vehicle("P0 MKT", mkt["C"].pct_change().dropna(), mkt, ff["RF"], pan["P0_market"]["target"], tuple(pan["P0_market"]["band"])); r0["panel"] = "P0"; res.append(r0); print("P0", flush=True)
    for t in TICKERS:
        px = load(t); r = run_vehicle(f"P1 {t}", px["C"].pct_change().dropna(), px, rf, pan["P1_1x"]["target"], tuple(pan["P1_1x"]["band"])); r["panel"] = "P1"; res.append(r); print(t, flush=True)
    for s, v in PAIRS.items():
        ps, pv = load(s), load(v); r = run_vehicle(f"P2 {v}", ps["C"].pct_change().dropna(), pv, rf, pan["P2_3x"]["target"], tuple(pan["P2_3x"]["band"])); r["panel"] = "P2"; res.append(r); print(v, flush=True)
    res = pd.concat(res, ignore_index=True); res.to_csv(OUT / "all_results.csv", index=False); report(res)


if __name__ == "__main__":
    main()
