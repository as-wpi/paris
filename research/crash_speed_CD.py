"""Crash speed — tests C (two-speed sizer) and D (shock rule).
Pre-registration: Strategy Research Memos/crash-speed-switch-vs-sizer/preregistration_CD.json.

    ~/.venvs/rtl-workspace/bin/python research/crash_speed_CD.py
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

PREREG = json.loads((MEMO / "preregistration_CD.json").read_text())
OUT = _HERE / "crash_speed_CD"
SHOCK_K, SHOCK_DAYS, SHOCK_FRAC = 3.0, 5, 0.5


def ewma_vol_lam(r, lam):
    x = np.log1p(r)
    v = (x**2).ewm(alpha=1 - lam, adjust=False).mean()
    v.iloc[:60] = np.nan
    return np.sqrt(v * 252)


def vt_shock(gate, sigma, r, rf, target, band, cap=1.0):
    """§14.1 loop + shock rule: after r_t < -K·sigma (sigma known at T-1), the held weight is
    halved for the next SHOCK_DAYS days (the halving is a hard cap on the target), then released."""
    n = len(r)
    w = np.zeros(n); ret = np.zeros(n); w_prev = 0.0; shock_until = -1
    g, s, rr, f = gate.to_numpy(), sigma.to_numpy(), r.to_numpy(), rf.to_numpy()
    for t in range(n):
        tgt = w_prev
        if not (g[t] == 1) or not np.isfinite(s[t]) or s[t] <= 0:
            tgt = 0.0
        else:
            w_star = min(target / s[t], cap)
            if w_prev == 0.0:
                tgt = w_star
            else:
                implied = w_prev * s[t]
                if implied < band[0] or implied > band[1]:
                    tgt = w_star
            if t <= shock_until:
                tgt = min(tgt, SHOCK_FRAC * w_star)
        cost = abs(tgt - w_prev) * COST
        ret[t] = tgt * rr[t] + (1 - tgt) * f[t] - cost
        w[t] = tgt
        gross = 1 + tgt * rr[t] + (1 - tgt) * f[t]
        w_prev = tgt * (1 + rr[t]) / gross if gross > 0 else 0.0
        # shock detected at close t (known for t+1 onward)
        if np.isfinite(s[t]) and s[t] > 0 and rr[t] < -SHOCK_K * s[t] / np.sqrt(252):
            shock_until = t + SHOCK_DAYS
    return pd.Series(w, index=r.index), pd.Series(ret, index=r.index)


def track_rmse(ret, target):
    roll = ret.rolling(63).std() * np.sqrt(252)
    return float(np.sqrt(((roll.dropna() - target) ** 2).mean()))


def run_vehicle(name, sig_r, veh, rf, target, band):
    rv = veh["C"].pct_change().dropna()
    trend0 = paris.trend_states(sig_r, jump_penalty=5.0, lag=0)
    idx = rv.index
    rf_ = rf.reindex(idx).ffill().fillna(0.0)
    g = trend0.reindex(idx).ffill().shift(1).fillna(0.0)
    ones = pd.Series(1.0, index=idx)
    s94 = ewma_vol(rv).shift(1).reindex(idx)
    sig = {"incumbent": s94,
           "C_85": pd.concat([s94, ewma_vol_lam(rv, 0.85).shift(1).reindex(idx)], axis=1).max(axis=1),
           "C_80": pd.concat([s94, ewma_vol_lam(rv, 0.80).shift(1).reindex(idx)], axis=1).max(axis=1)}
    arms = {k: vt_path(g, s, rv, rf_, target, band) for k, s in sig.items()}
    bh_ret = rv.copy(); bh_ret.iloc[0] -= COST
    arms["buy_hold"] = (ones, bh_ret)
    arms["D_shock"] = vt_shock(g, s94, rv, rf_, target, band)
    for k in ("C_85", "C_80"):
        arms[f"D_shock_on_{k}"] = vt_shock(g, sig[k], rv, rf_, target, band)
    ungated = {k: vt_path(ones, s, rv, rf_, target, band)[1] for k, s in sig.items()}
    start = max(trend0.reindex(idx).first_valid_index(), max(v[1].first_valid_index() for v in arms.values()))
    rows = []
    for label, (w, ret) in arms.items():
        ret, w = ret.loc[start:], w.loc[start:]
        rows.append({"vehicle": name, "arm": label, "episode": "FULL", "Sharpe": float(paris.sharpe(ret, rf=rf_.loc[ret.index])),
                     "Martin": float(paris.martin_ratio(ret)), "MaxDD": float(paris.max_drawdown(ret)),
                     "Turnover/yr": float(w.diff().abs().sum() / (len(w) / 252)),
                     "TrackRMSE": track_rmse(ungated[label if label in ungated else "incumbent"].loc[start:], target) if label in ungated else np.nan})
        for ek, (a, b, grp) in episodes().items():
            if a < start:
                continue
            st = ep_stats(ret, w, a, b)
            if st:
                seg = ret.loc[a:b]
                st["Sharpe"] = float(paris.sharpe(seg, rf=rf_.loc[seg.index]))
                rows.append({"vehicle": name, "arm": label, "episode": ek, "group": grp, **st})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    rf = load_rf(); ff = load_ff(str(FF))
    pan = json.loads((MEMO / "preregistration_AB.json").read_text())["stack_under_study"]["panels"]
    res = []
    mkt = pd.DataFrame({"C": (1 + ff["MKT"].loc["1926":]).cumprod()})
    r0 = run_vehicle("P0 MKT", mkt["C"].pct_change().dropna(), mkt, ff["RF"], pan["P0_market"]["target"], tuple(pan["P0_market"]["band"])); r0["panel"] = "P0"; res.append(r0); print("P0 done", flush=True)
    for t in TICKERS:
        px = load(t); r = run_vehicle(f"P1 {t}", px["C"].pct_change().dropna(), px, rf, pan["P1_1x"]["target"], tuple(pan["P1_1x"]["band"])); r["panel"] = "P1"; res.append(r); print(t, flush=True)
    for s, v in PAIRS.items():
        ps, pv = load(s), load(v); r = run_vehicle(f"P2 {v}", ps["C"].pct_change().dropna(), pv, rf, pan["P2_3x"]["target"], tuple(pan["P2_3x"]["band"])); r["panel"] = "P2"; res.append(r); print(v, flush=True)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(OUT / "all_results.csv", index=False)
    report(res)


def report(res):
    rng = np.random.default_rng(0)
    full = res[res.episode == "FULL"]; ep = res[res.episode != "FULL"]
    calm = ["K1995_99_bull", "K2013_14_bull", "A_bull_2021", "D_recent_2025plus"]
    L = [f"# Crash speed — tests C and D — {pd.Timestamp.today().date()}\n", "Pre-registration: `preregistration_CD.json`. Ceiling from test A: sizer-only k=5 bound = 25% of fast-crash MaxDD.\n"]
    L.append("## Full sample (medians across 16 vehicles; Δ paired vs incumbent)\n\n| Arm | Sharpe | Martin | MaxDD | Turnover/yr | Track RMSE (ungated) |\n|---|---:|---:|---:|---:|---:|")
    for arm in ("incumbent", "C_85", "C_80", "D_shock", "D_shock_on_C_85", "D_shock_on_C_80"):
        m = full[full.arm == arm][["Sharpe", "Martin", "MaxDD", "Turnover/yr", "TrackRMSE"]].median()
        L.append(f"| {arm} | {m.Sharpe:.2f} | {m.Martin:.2f} | {m.MaxDD:.1%} | {m['Turnover/yr']:.1f} | {'' if np.isnan(m.TrackRMSE) else f'{m.TrackRMSE:.1%}'} |")
    L.append("\n## Fast-crash episodes — median MaxDD by arm\n\n| Episode | incumbent | C_85 | C_80 | D_shock | D_shock_on_C_85 | D_shock_on_C_80 |\n|---|---:|---:|---:|---:|---:|---:|")
    for e in FAST + calm:
        row = [ep[(ep.episode == e) & (ep.arm == a)]["MaxDD"].median() for a in ("incumbent", "C_85", "C_80", "D_shock", "D_shock_on_C_85", "D_shock_on_C_80")]
        L.append(f"| {e} | " + " | ".join("—" if np.isnan(x) else f"{x:.1%}" for x in row) + " |")
    L.append("\n## Pre-registered decision rule\n")
    verdict = {}
    for arm in ("C_85", "C_80", "D_shock", "D_shock_on_C_85", "D_shock_on_C_80"):
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
        bootd = []
        tr = (full.pivot(index="vehicle", columns="arm", values="Turnover/yr")[arm] / full.pivot(index="vehicle", columns="arm", values="Turnover/yr")["incumbent"]).median()
        # registered bleed clause (preregistration_CD.json): per calm control, cross-vehicle median of
        # (B&H Sharpe - arm Sharpe) must not exceed (B&H Sharpe - incumbent Sharpe) + 0.10
        bleed_ok, notes = True, []
        for e in calm:
            sub = ep[ep.episode == e].pivot(index="vehicle", columns="arm", values="Sharpe")
            if arm not in sub.columns or "buy_hold" not in sub.columns:
                continue
            bl_a = (sub["buy_hold"] - sub[arm]).dropna().median()
            bl_i = (sub["buy_hold"] - sub["incumbent"]).dropna().median()
            ok = bl_a <= bl_i + 0.10
            bleed_ok &= bool(ok)
            notes.append(f"{e}: bleed vs B&H {bl_a:+.2f} (arm) vs {bl_i:+.2f} (incumbent) {'ok' if ok else 'FAIL'}")
        ok_all = (ddm >= 0.02) and (ds.median() >= -0.02) and bleed_ok and (tr <= 1.25)
        verdict[arm] = "PASS" if ok_all else "RETIRE"
        L.append(f"\n**{arm}: {verdict[arm]}** — fast-crash ΔMaxDD {ddm*100:+.1f} pts (bar +2.0); ΔSharpe {ds.median():+.3f} (CI [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}], bar −0.02); turnover ×{tr:.2f} (bar 1.25); calm-control bleed {'pass' if bleed_ok else 'FAIL'}.")
        for n_ in notes:
            L.append(f"- {n_}")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report(pd.read_csv(OUT / "all_results.csv"))
    else:
        main()
