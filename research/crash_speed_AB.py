"""Crash speed — tests A (attribution bound) and B (execution timing).
Pre-registration: ~/Documents/Work/Strategy Research Memos/crash-speed-switch-vs-sizer/preregistration_AB.json
(incl. the outcome-blind amendment restating B).

    ~/.venvs/rtl-workspace/bin/python research/crash_speed_AB.py
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
from frozen_rule_benchmark import DATA, FF, TICKERS, load_rf  # noqa: E402
from walkforward_jump import load_ff  # noqa: E402

MEMO = Path.home() / "Documents/Work/Strategy Research Memos/crash-speed-switch-vs-sizer"
PREREG = json.loads((MEMO / "preregistration_AB.json").read_text())
OUT = _HERE / "crash_speed_AB"
COST = (5.0 + 3.0) / 1e4
LAM, WARM = 0.94, 60
PAIRS = PREREG["stack_under_study"]["panels"]["P2_3x"]["pairs"]
FAST = ["E1987_crash", "E1998_ltcm", "X_2020_covid", "E_bear_2018q4"]


def load(t):
    df = pd.read_csv(DATA / f"{t}_adj.csv", index_col=0, parse_dates=True)
    return df.rename(columns={"adjOpen": "O", "adjClose": "C"})[["O", "C"]].dropna()


def ewma_vol(r):
    x = np.log1p(r)
    v = (x**2).ewm(alpha=1 - LAM, adjust=False).mean()
    v.iloc[:WARM] = np.nan
    return np.sqrt(v * 252)


def vt_path(gate, sigma, r, rf, target, band, cap=1.0, r_open=None, r_gap=None, delay=0):
    """§14.1 loop. gate/sigma indexed at the day they ACT on (already shifted by the caller).
    delay=1: the decided weight takes effect one day later (t+2). r_open/r_gap: next-open route —
    day-T return = old weight on the overnight gap (close T-1 -> open T) + new weight open -> close."""
    n = len(r)
    w = np.zeros(n); ret = np.zeros(n); w_prev = 0.0; pending = None
    g, s, rr, f = gate.to_numpy(), sigma.to_numpy(), r.to_numpy(), rf.to_numpy()
    ro = r_open.to_numpy() if r_open is not None else None
    rg = r_gap.to_numpy() if r_gap is not None else None
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
        if delay:
            # the decision made for t executes at close t and earns from t+1
            act = pending if pending is not None else w_prev
            pending = tgt
            cost = abs(act - w_prev) * COST
            ret[t] = act * rr[t] + (1 - act) * f[t] - cost
            w[t] = act
            gross = 1 + act * rr[t] + (1 - act) * f[t]
            w_prev = act * (1 + rr[t]) / gross if gross > 0 else 0.0
            continue
        cost = abs(tgt - w_prev) * COST
        if ro is not None:
            # next-open: the old weight rides the overnight gap (cash earns rf), the new weight
            # rides open -> close; the held weight drifts with the open -> close move only
            overnight = 1 + w_prev * rg[t] + (1 - w_prev) * f[t]
            intraday = 1 + tgt * ro[t]
            ret[t] = overnight * intraday - 1 - cost
            w[t] = tgt
            w_prev = tgt * (1 + ro[t]) / intraday if intraday > 0 else 0.0
            continue
        ret[t] = tgt * rr[t] + (1 - tgt) * f[t] - cost
        w[t] = tgt
        gross = 1 + tgt * rr[t] + (1 - tgt) * f[t]
        w_prev = tgt * (1 + rr[t]) / gross if gross > 0 else 0.0
    return pd.Series(w, index=r.index), pd.Series(ret, index=r.index)


def episodes():
    ep = PREREG["episodes"]
    out = {}
    for grp in ("modern_folds", "market_factor_A0_calendar", "calm_controls"):
        for k, (a, b) in ep[grp].items():
            out[k] = (pd.Timestamp(a), pd.Timestamp(b), grp)
    return out


def ep_stats(ret: pd.Series, w: pd.Series, a, b):
    s = ret.loc[a:b]
    if len(s) < 20:
        return None
    eq = (1 + s).cumprod()
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    ww = w.loc[a:b]
    w_peak = float(ww.loc[peak]) if peak in ww.index else np.nan
    react = ww.loc[peak:trough]
    hit = react[react < 0.5 * w_peak].index.min() if w_peak and np.isfinite(w_peak) and w_peak > 0 else pd.NaT
    dd_before = float(eq.loc[peak:hit].min() / eq.loc[peak] - 1) if pd.notna(hit) else float(dd.min())
    return {"MaxDD": float(dd.min()), "Return": float(eq.iloc[-1] - 1), "peak": peak.date(), "trough": trough.date(),
            "react_day": (hit.date() if pd.notna(hit) else None), "DD_before_reaction": dd_before,
            "days_peak_to_trough": int(len(eq.loc[peak:trough])), "w_at_peak": w_peak}


def run_vehicle(name, sig_r, veh, rf, target, band, has_open):
    """sig_r: 1x returns for the switch; veh: DataFrame with C (and O) of the vehicle."""
    rv = veh["C"].pct_change().dropna()
    trend0 = paris.trend_states(sig_r, jump_penalty=5.0, lag=0)          # state at close T
    sig0 = ewma_vol(rv)                                                   # sigma at close T
    idx = rv.index
    rf_ = rf.reindex(idx).ffill().fillna(0.0)
    arms = {}
    # test A: incumbent and lookahead bounds (fiction)
    for k_sw, k_sz, label in ((1, 1, "incumbent t+1"), (0, 0, "A: both k=1"), (0, 1, "A: switch k=1"), (1, 0, "A: sizer k=1"),
                              (-4, -4, "A: both k=5"), (-4, 1, "A: switch k=5"), (1, -4, "A: sizer k=5")):
        g = trend0.reindex(idx).ffill().shift(k_sw).fillna(0.0)
        s = sig0.shift(k_sz).reindex(idx)
        arms[label] = vt_path(g, s, rv, rf_, target, band)
    # test B: execution timing
    g1 = trend0.reindex(idx).ffill().shift(1).fillna(0.0); s1 = sig0.shift(1).reindex(idx)
    arms["B: t+2 next-close"] = vt_path(g1, s1, rv, rf_, target, band, delay=1)
    if has_open:
        r_gap = (veh["O"] / veh["C"].shift(1) - 1).reindex(idx)
        r_oc = (veh["C"] / veh["O"] - 1).reindex(idx)
        arms["B: next-open"] = vt_path(g1, s1, rv, rf_, target, band, r_open=r_oc, r_gap=r_gap)
    start = max(a[1].first_valid_index() for a in arms.values() if a[1].first_valid_index() is not None)
    first_ok = trend0.reindex(idx).first_valid_index()
    start = max(start, first_ok)
    rows = []
    for label, (w, ret) in arms.items():
        ret, w = ret.loc[start:], w.loc[start:]
        rows.append({"vehicle": name, "arm": label, "episode": "FULL", "Sharpe": float(paris.sharpe(ret, rf=rf_.loc[ret.index])),
                     "Martin": float(paris.martin_ratio(ret)), "MaxDD": float(paris.max_drawdown(ret)), "Turnover/yr": float(w.diff().abs().sum() / (len(w) / 252))})
        for ek, (a, b, grp) in episodes().items():
            if a < start:
                continue
            st = ep_stats(ret, w, a, b)
            if st:
                rows.append({"vehicle": name, "arm": label, "episode": ek, "group": grp, **st})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv and (OUT / "all_results.csv").exists():
        report(pd.read_csv(OUT / "all_results.csv"))
        return
    rf = load_rf()
    ff = load_ff(str(FF))
    res = []
    p1 = PREREG["stack_under_study"]["panels"]["P1_1x"]; p2 = PREREG["stack_under_study"]["panels"]["P2_3x"]; p0 = PREREG["stack_under_study"]["panels"]["P0_market"]
    mkt = pd.DataFrame({"C": (1 + ff["MKT"].loc["1926":]).cumprod()})
    r0 = run_vehicle("P0 MKT", mkt["C"].pct_change().dropna(), mkt, ff["RF"], p0["target"], tuple(p0["band"]), has_open=False); r0["panel"] = "P0"; res.append(r0)
    print("P0 done", flush=True)
    for t in TICKERS:
        px = load(t); r = run_vehicle(f"P1 {t}", px["C"].pct_change().dropna(), px, rf, p1["target"], tuple(p1["band"]), True); r["panel"] = "P1"; res.append(r); print(t, "done", flush=True)
    for s, v in PAIRS.items():
        ps, pv = load(s), load(v); r = run_vehicle(f"P2 {v}", ps["C"].pct_change().dropna(), pv, rf, p2["target"], tuple(p2["band"]), True); r["panel"] = "P2"; res.append(r); print(v, "done", flush=True)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(OUT / "all_results.csv", index=False)
    report(res)


def report(res):
    L = [f"# Crash speed — tests A and B — {pd.Timestamp.today().date()}\n", "Pre-registration (with the outcome-blind B amendment): `Strategy Research Memos/crash-speed-switch-vs-sizer/`.\n"]
    ep = res[res.episode != "FULL"]
    inc = ep[ep.arm == "incumbent t+1"]
    L.append("\n## Test A, step 1 — where the drawdown accrues (incumbent stack; medians across vehicles)\n")
    L.append("| Episode | n | MaxDD | DD before sizer reaction | share before | days peak→trough | w at peak |\n|---|---:|---:|---:|---:|---:|---:|")
    for ek in episodes():
        sub = inc[inc.episode == ek]
        if sub.empty:
            continue
        share = (sub["DD_before_reaction"] / sub["MaxDD"]).replace([np.inf, -np.inf], np.nan)
        L.append(f"| {ek} | {len(sub)} | {sub.MaxDD.median():.1%} | {sub.DD_before_reaction.median():.1%} | {share.median():.0%} | {sub.days_peak_to_trough.median():.0f} | {sub.w_at_peak.median():.2f} |")
    L.append("\n## Test A, step 2 — lookahead bounds (FICTION): reachable share of MaxDD = (bound − actual)/|actual|, medians (positive = the bound loses less)\n")
    L.append("| Episode | both k=1 | switch k=1 | sizer k=1 | both k=5 | switch k=5 | sizer k=5 |\n|---|---:|---:|---:|---:|---:|---:|")
    shares = {}
    for ek in episodes():
        row = []
        for arm in ("A: both k=1", "A: switch k=1", "A: sizer k=1", "A: both k=5", "A: switch k=5", "A: sizer k=5"):
            a = ep[(ep.episode == ek) & (ep.arm == arm)].set_index("vehicle")["MaxDD"]
            i = inc[inc.episode == ek].set_index("vehicle")["MaxDD"]
            d = ((a.reindex(i.index) - i) / i.abs()).dropna()   # bound is less negative -> positive share
            shares[(ek, arm)] = float(d.median()) if len(d) else np.nan
            row.append(shares[(ek, arm)])
        if all(np.isnan(row)):
            continue
        L.append(f"| {ek} | " + " | ".join("—" if np.isnan(x) else f"{x:+.0%}" for x in row) + " |")
    fast_med = {arm: np.nanmedian([shares.get((e, arm), np.nan) for e in FAST]) for arm in ("A: both k=5", "A: switch k=5", "A: sizer k=5", "A: both k=1")}
    L.append(f"\n**Fast-crash subset ({', '.join(FAST)}) medians:** both k=5 {fast_med['A: both k=5']:+.0%}, switch-only k=5 {fast_med['A: switch k=5']:+.0%}, sizer-only k=5 {fast_med['A: sizer k=5']:+.0%}; both k=1 {fast_med['A: both k=1']:+.0%}.")
    both, sw, sz = fast_med["A: both k=5"], fast_med["A: switch k=5"], fast_med["A: sizer k=5"]
    if both < 1 / 3:
        verdict_a = "SPEED IS NOT THE LEVER — only H (drawdown governor) proceeds"
    elif sz >= 2 / 3 * both:
        verdict_a = "SIZER CARRIES IT — E, F, G dropped; C, D proceed"
    elif sw >= 2 / 3 * both:
        verdict_a = "SWITCH CARRIES IT — C, D dropped; E, F, G proceed"
    else:
        verdict_a = "MIXED — C–G proceed in the registered order"
    L.append(f"\n**Test A verdict (pre-registered rule): {verdict_a}.**")
    # B
    L.append("\n## Test B — execution timing (full sample, paired vs the incumbent; fast-crash MaxDD)\n")
    full = res[res.episode == "FULL"]
    rng = np.random.default_rng(0)
    L.append("| Arm | ΔSharpe median (CI) | ΔMartin | fast-crash ΔMaxDD (pts) | turnover × |\n|---|---:|---:|---:|---:|")
    b_res = {}
    for arm in ("B: next-open", "B: t+2 next-close", "A: both k=1"):
        pv = full.pivot(index="vehicle", columns="arm", values="Sharpe")
        if arm not in pv.columns:
            continue
        d = (pv[arm] - pv["incumbent t+1"]).dropna()
        boot = [np.median(rng.choice(d.values, len(d))) for _ in range(1000)]
        dm = (full.pivot(index="vehicle", columns="arm", values="Martin")[arm] - full.pivot(index="vehicle", columns="arm", values="Martin")["incumbent t+1"]).median()
        fc = ep[ep.episode.isin(FAST)]
        dd = []
        for e in FAST:
            a = fc[(fc.episode == e) & (fc.arm == arm)].set_index("vehicle")["MaxDD"]; i = fc[(fc.episode == e) & (fc.arm == "incumbent t+1")].set_index("vehicle")["MaxDD"]
            x = (a.reindex(i.index) - i).dropna()
            if len(x):
                dd.append(float(x.median()))
        ddm = float(np.median(dd)) if dd else np.nan
        tr = (full.pivot(index="vehicle", columns="arm", values="Turnover/yr")[arm] / full.pivot(index="vehicle", columns="arm", values="Turnover/yr")["incumbent t+1"]).median()
        b_res[arm] = (float(d.median()), ddm)
        L.append(f"| {arm} | {d.median():+.3f} ([{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}]) | {dm:+.2f} | {ddm*100:+.1f} | {tr:.2f} |")
    t2 = b_res.get("B: t+2 next-close", (0, 0)); no = b_res.get("B: next-open", (0, 0))
    lag_matters = (t2[1] <= -0.02) or (t2[0] <= -0.03)
    open_recovers = not ((no[1] <= -0.02) or (no[0] <= -0.03))
    verdict_b = ("MOC ROUTE WORTH ITS COST" if (lag_matters and not open_recovers) else ("NEXT-OPEN SUFFICES" if lag_matters else "EXECUTION TIMING IS NOT A LEVER"))
    L.append(f"\n**Test B verdict (pre-registered rule): {verdict_b}** — t+2 ΔSharpe {t2[0]:+.3f}, fast-crash ΔMaxDD {t2[1]*100:+.1f} pts; next-open ΔSharpe {no[0]:+.3f}, ΔMaxDD {no[1]*100:+.1f} pts.")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    (OUT / "verdict.json").write_text(json.dumps({"A": verdict_a, "B": verdict_b, "fast_med": fast_med, "B": b_res}, indent=2, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
