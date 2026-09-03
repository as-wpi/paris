"""Crash speed — tests E (circuit breaker), F (asymmetric penalty), G (turbulence exit).
Pre-registration: Strategy Research Memos/crash-speed-switch-vs-sizer/preregistration_EFG.json.

    ~/.venvs/rtl-workspace/bin/python research/crash_speed_EFG.py
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
from paris.jump import _loss, _standardise, _unpack_fits  # noqa: E402
from crash_speed_AB import COST, FAST, MEMO, PAIRS, ep_stats, episodes, ewma_vol, load, vt_path  # noqa: E402
from frozen_rule_benchmark import FF, TICKERS, load_rf  # noqa: E402
from walkforward_jump import load_ff  # noqa: E402

OUT = _HERE / "crash_speed_EFG"
CALM = ["K1995_99_bull", "K2013_14_bull", "A_bull_2021", "D_recent_2025plus"]
WINDOW, LAM_ON = 1260, 5.0


def asymmetric_states(r: pd.Series, lam_off: float) -> pd.Series:
    """Lag-0 online states with the incumbent's fitted centres and an asymmetric penalty:
    lam_on (off -> on) = 5, lam_off (on -> off) = lam_off. Forward values only."""
    slow, fast = paris.trend_signal(r, "slow"), paris.trend_signal(r, "fast")
    feat = pd.DataFrame({"slow": slow, "fast": fast}).dropna()
    fits = paris.trend_fits(r, jump_penalty=LAM_ON, window=WINDOW)
    packed = _unpack_fits(fits, feat, 2)
    X = feat.to_numpy(dtype=float)
    n = len(feat)
    out = np.full(n, np.nan)
    refit_dates = sorted(packed)
    starts = [feat.index.get_loc(d) for d in refit_dates]
    for j, (d, start) in enumerate(zip(refit_dates, starts)):
        mu, sd, centers = packed[d]
        stop = starts[j + 1] if j + 1 < len(starts) else n
        for t in range(start, stop):
            lo = max(0, t - WINDOW + 1)
            L = _loss(_standardise(X[lo:t + 1], mu, sd, 3.0), centers)
            v0, v1 = float(L[0, 0]), float(L[0, 1])
            for i in range(1, len(L)):
                n0 = L[i, 0] + min(v0, v1 + lam_off)     # into state 0 (off): stay, or exit from 1
                n1 = L[i, 1] + min(v1, v0 + LAM_ON)      # into state 1 (on): stay, or enter from 0
                m = min(n0, n1); v0, v1 = n0 - m, n1 - m
            out[t] = 0.0 if v0 <= v1 else 1.0
    return pd.Series(out, index=feat.index)


def turbulence_flag(r: pd.Series) -> pd.Series:
    """1 when the 5-day mean of (r/sigma_daily,T-1)^2 exceeds its trailing 1,260-day 95th pct (both at T-1)."""
    sig_d = (ewma_vol(r) / np.sqrt(252)).shift(1)
    z2 = (r / sig_d) ** 2
    turb = z2.rolling(5).mean()
    thr = turb.rolling(WINDOW, min_periods=WINDOW).quantile(0.95)
    return ((turb > thr).astype(float).where(turb.notna() & thr.notna())).shift(1)


def breaker_gate(gate: pd.Series, sig_r: pd.Series, k: float) -> pd.Series:
    """gate already lagged to the day it acts on; trip when the 21-day return at T-1 < -k sigma_21."""
    ret21 = sig_r.rolling(21).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    sig21 = ewma_vol(sig_r) * np.sqrt(21 / 252)
    trip = (ret21 < -k * sig21).shift(1).reindex(gate.index).fillna(False).astype(bool).to_numpy()
    g = gate.to_numpy().copy()
    until = -1
    for t in range(len(g)):
        if trip[t]:
            until = t + 21
        if t <= until:
            g[t] = 0.0
    return pd.Series(g, index=gate.index)


def run_vehicle(name, sig_r, veh, rf, target, band, spy_r=None):
    rv = veh["C"].pct_change().dropna()
    idx = rv.index
    rf_ = rf.reindex(idx).ffill().fillna(0.0)
    trend0 = paris.trend_states(sig_r, jump_penalty=5.0, lag=0)
    g_inc = trend0.reindex(idx).ffill().shift(1).fillna(0.0)
    s94 = ewma_vol(rv).shift(1).reindex(idx)
    gates = {"incumbent": g_inc,
             "E_2s": breaker_gate(g_inc, sig_r, 2.0), "E_3s": breaker_gate(g_inc, sig_r, 3.0),
             "F_exit2": asymmetric_states(sig_r, 2.0).reindex(idx).ffill().shift(1).fillna(0.0),
             "F_exit1": asymmetric_states(sig_r, 1.0).reindex(idx).ffill().shift(1).fillna(0.0)}
    arms = {k: vt_path(g, s94, rv, rf_, target, band) for k, g in gates.items()}
    # G: exposure x 0.5 while turbulent — implemented as a cap on the held weight (sizer otherwise unchanged)
    for k, src in (("G_own", sig_r), ("G_spy", spy_r if spy_r is not None else sig_r)):
        flag = turbulence_flag(src).reindex(idx).ffill().fillna(0.0)
        w, ret = arms["incumbent"]
        w2 = w * np.where(flag.to_numpy() == 1.0, 0.5, 1.0)
        cost = np.abs(np.diff(np.concatenate([[0.0], w2.to_numpy()]))) * COST
        ret2 = w2 * rv + (1 - w2) * rf_ - cost
        arms[k] = (w2, ret2)
    bh = rv.copy(); bh.iloc[0] -= COST
    arms["buy_hold"] = (pd.Series(1.0, index=idx), bh)
    start = max(trend0.reindex(idx).first_valid_index(), max(v[1].first_valid_index() for v in arms.values()))
    rows = []
    for label, (w, ret) in arms.items():
        ret, w = ret.loc[start:], w.loc[start:]
        g = gates.get(label)
        flips = float((g.loc[start:].diff().abs() > 0).sum() / (len(w) / 252)) if g is not None else np.nan
        rows.append({"vehicle": name, "arm": label, "episode": "FULL", "Sharpe": float(paris.sharpe(ret, rf=rf_.loc[ret.index])),
                     "Martin": float(paris.martin_ratio(ret)), "MaxDD": float(paris.max_drawdown(ret)),
                     "Turnover/yr": float(w.diff().abs().sum() / (len(w) / 252)), "Flips/yr": flips})
        for ek, (a, b, grp) in episodes().items():
            if a < start:
                continue
            st = ep_stats(ret, w, a, b)
            if st:
                seg = ret.loc[a:b]
                st["Sharpe"] = float(paris.sharpe(seg, rf=rf_.loc[seg.index]))
                rows.append({"vehicle": name, "arm": label, "episode": ek, "group": grp, **st})
    return pd.DataFrame(rows)


ARMS = ["E_2s", "E_3s", "F_exit2", "F_exit1", "G_own", "G_spy"]


def report(res):
    rng = np.random.default_rng(0)
    full = res[res.episode == "FULL"]; ep = res[res.episode != "FULL"]
    L = [f"# Crash speed — tests E, F, G — {pd.Timestamp.today().date()}\n", "Pre-registration: `preregistration_EFG.json`. Ceiling from test A: switch-only k=5 bound = 16% of fast-crash MaxDD (~1 pt).\n"]
    L.append("## Full sample (medians across 16 vehicles)\n\n| Arm | Sharpe | Martin | MaxDD | Turnover/yr | Flips/yr |\n|---|---:|---:|---:|---:|---:|")
    for arm in ["incumbent"] + ARMS:
        m = full[full.arm == arm][["Sharpe", "Martin", "MaxDD", "Turnover/yr", "Flips/yr"]].median()
        L.append(f"| {arm} | {m.Sharpe:.2f} | {m.Martin:.2f} | {m.MaxDD:.1%} | {m['Turnover/yr']:.1f} | {'' if np.isnan(m['Flips/yr']) else f'{m['Flips/yr']:.1f}'} |")
    L.append("\n## Fast-crash and calm episodes — median MaxDD by arm\n\n| Episode | incumbent | " + " | ".join(ARMS) + " |\n|---|" + "---:|" * (len(ARMS) + 1))
    for e in FAST + CALM:
        row = [ep[(ep.episode == e) & (ep.arm == a)]["MaxDD"].median() for a in ["incumbent"] + ARMS]
        L.append(f"| {e} | " + " | ".join("—" if np.isnan(x) else f"{x:.1%}" for x in row) + " |")
    L.append("\n## Pre-registered decision rule\n")
    verdict = {}
    for arm in ARMS:
        pv = full.pivot(index="vehicle", columns="arm", values="Sharpe")
        ds = (pv[arm] - pv["incumbent"]).dropna()
        boot = [np.median(rng.choice(ds.values, len(ds))) for _ in range(1000)]
        dd = []
        for e in FAST:
            a = ep[(ep.episode == e) & (ep.arm == arm)].set_index("vehicle")["MaxDD"]; i = ep[(ep.episode == e) & (ep.arm == "incumbent")].set_index("vehicle")["MaxDD"]
            x = (a.reindex(i.index) - i).dropna()
            if len(x):
                dd.append(float(x.median()))
        ddm = float(np.median(dd)) if dd else np.nan
        tr = (full.pivot(index="vehicle", columns="arm", values="Turnover/yr")[arm] / full.pivot(index="vehicle", columns="arm", values="Turnover/yr")["incumbent"]).median()
        fl = full[full.arm == arm]["Flips/yr"].median()
        bleed_ok, notes = True, []
        for e in CALM:
            sub = ep[ep.episode == e].pivot(index="vehicle", columns="arm", values="Sharpe")
            if arm not in sub.columns:
                continue
            bl_a = (sub["buy_hold"] - sub[arm]).dropna().median(); bl_i = (sub["buy_hold"] - sub["incumbent"]).dropna().median()
            ok = bl_a <= bl_i + 0.10
            bleed_ok &= bool(ok)
            notes.append(f"{e}: bleed vs B&H {bl_a:+.2f} (arm) vs {bl_i:+.2f} (incumbent) {'ok' if ok else 'FAIL'}")
        flips_ok = (np.isnan(fl)) or (fl <= 4.0)
        ok_all = (ddm >= 0.02) and (ds.median() >= -0.02) and bleed_ok and (tr <= 1.25) and flips_ok
        verdict[arm] = "PASS" if ok_all else "RETIRE"
        L.append(f"\n**{arm}: {verdict[arm]}** — fast-crash ΔMaxDD {ddm*100:+.1f} pts (bar +2.0; ceiling ~+1.0); ΔSharpe {ds.median():+.3f} (CI [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}], bar −0.02); turnover ×{tr:.2f} (bar 1.25); flips/yr {'n/a' if np.isnan(fl) else f'{fl:.1f}'} (bar 4); calm-control bleed {'pass' if bleed_ok else 'FAIL'}.")
        for n_ in notes:
            L.append(f"- {n_}")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n".join(L))


def main():
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        report(pd.read_csv(OUT / "all_results.csv")); return
    rf = load_rf(); ff = load_ff(str(FF))
    pan = json.loads((MEMO / "preregistration_AB.json").read_text())["stack_under_study"]["panels"]
    spy_r = load("SPY")["C"].pct_change().dropna()
    res = []
    mkt = pd.DataFrame({"C": (1 + ff["MKT"].loc["1926":]).cumprod()})
    r0 = run_vehicle("P0 MKT", mkt["C"].pct_change().dropna(), mkt, ff["RF"], pan["P0_market"]["target"], tuple(pan["P0_market"]["band"])); r0["panel"] = "P0"; res.append(r0); print("P0 done", flush=True)
    for t in TICKERS:
        px = load(t); r = run_vehicle(f"P1 {t}", px["C"].pct_change().dropna(), px, rf, pan["P1_1x"]["target"], tuple(pan["P1_1x"]["band"]), spy_r); r["panel"] = "P1"; res.append(r); print(t, flush=True)
    for s, v in PAIRS.items():
        ps, pv = load(s), load(v); r = run_vehicle(f"P2 {v}", ps["C"].pct_change().dropna(), pv, rf, pan["P2_3x"]["target"], tuple(pan["P2_3x"]["band"]), spy_r); r["panel"] = "P2"; res.append(r); print(v, flush=True)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(OUT / "all_results.csv", index=False)
    report(res)


if __name__ == "__main__":
    main()
