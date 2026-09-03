"""B1 as a sizing-estimator test: volatility estimators inside the vol-target loop.
Pre-registration: research/sizing_estimator/preregistration.json.

    ~/.venvs/rtl-workspace/bin/python research/sizing_estimator.py --calibration rolling|expanding
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WORK = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_WORK / "Quant Science" / "projects" / "RTL-Kalman" / "src"))
import paris  # noqa: E402
from frozen_rule_benchmark import DATA, TICKERS, load_rf  # noqa: E402
from rtl_kalman.indicators.kalman import kalman_local_linear_trend  # noqa: E402

OUT = _HERE / "sizing_estimator"
PREREG = json.loads((OUT / "preregistration.json").read_text())
PAIRS = PREREG["panels"]["P2_3x"]["pairs"]
COST = (5.0 + 3.0) / 1e4
LAM, WARM = 0.94, 60
RANGE_FACTOR = 2.0 * np.sqrt(2.0 / np.pi)  # E[H-L] / sigma for driftless Brownian motion


def load(t: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{t}_adj.csv", index_col=0, parse_dates=True)
    return df.rename(columns={"adjOpen": "O", "adjHigh": "H", "adjLow": "L", "adjClose": "C"})[["O", "H", "L", "C"]].dropna()


# ---------------------------------------------------------------- estimators (annualised, at close T)
def e0_ewma(px: pd.DataFrame) -> pd.Series:
    x = np.log(px["C"]).diff()
    v = (x**2).ewm(alpha=1 - LAM, adjust=False).mean()
    v.iloc[:WARM] = np.nan
    return np.sqrt(v * 252)


def e1_yang_zhang(px: pd.DataFrame, n: int = 21) -> pd.Series:
    o = np.log(px["O"] / px["C"].shift(1))
    c = np.log(px["C"] / px["O"])
    u = np.log(px["H"] / px["O"])
    d = np.log(px["L"] / px["O"])
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    vo = o.rolling(n).var()
    vc = c.rolling(n).var()
    rs = (u * (u - c) + d * (d - c)).rolling(n).mean()
    return np.sqrt((vo + k * vc + (1 - k) * rs).clip(lower=0) * 252)


def e2_parkinson(px: pd.DataFrame) -> pd.Series:
    p = np.log(px["H"] / px["L"]) ** 2 / (4 * np.log(2))
    v = p.ewm(alpha=1 - LAM, adjust=False).mean()
    v.iloc[:WARM] = np.nan
    return np.sqrt(v * 252)


def e3_katr(px: pd.DataFrame) -> pd.Series:
    h, l, c = px["H"].to_numpy(), px["L"].to_numpy(), px["C"].to_numpy()
    pc = np.roll(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    tr_pct = tr / c * 100.0
    level, *_ = kalman_local_linear_trend(np.log(np.maximum(tr_pct, 1e-8)), 1e-4, 1e-6, 1e-3)
    daily_sigma = np.exp(level) / 100.0 / RANGE_FACTOR
    out = pd.Series(daily_sigma * np.sqrt(252), index=px.index)
    out.iloc[:WARM] = np.nan
    return out


def e4_yz_ewma(px: pd.DataFrame, n: int = 21) -> pd.Series:
    """Yang-Zhang components with the incumbent's exponential filter (amendment E4)."""
    o = np.log(px["O"] / px["C"].shift(1))
    c = np.log(px["C"] / px["O"])
    u = np.log(px["H"] / px["O"])
    d = np.log(px["L"] / px["O"])
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    ew = lambda x: x.ewm(alpha=1 - LAM, adjust=False).mean()  # noqa: E731
    vo = ew(o**2) - ew(o) ** 2
    vc = ew(c**2) - ew(c) ** 2
    rs = ew(u * (u - c) + d * (d - c))
    v = (vo + k * vc + (1 - k) * rs).clip(lower=0)
    v.iloc[:WARM] = np.nan
    return np.sqrt(v * 252)


ESTIMATORS = {"E0 EWMA c2c": e0_ewma, "E1 Yang-Zhang 21": e1_yang_zhang, "E2 Parkinson EWMA": e2_parkinson, "E3 K-ATR fast": e3_katr,
              "E4 Yang-Zhang EWMA": e4_yz_ewma}


# ---------------------------------------------------------------- vol-target loop (§14.1 construction)
def run(gate: pd.Series, r: pd.Series, rf: pd.Series, sigma: pd.Series, target: float, band: tuple[float, float],
        cap: float) -> tuple[pd.Series, pd.Series]:
    n = len(r)
    w = np.zeros(n)
    ret = np.zeros(n)
    w_prev = 0.0
    g, s, rr, f = gate.to_numpy(), sigma.to_numpy(), r.to_numpy(), rf.to_numpy()
    for t in range(n):
        tgt = w_prev
        if not (g[t] == 1) or np.isnan(s[t]):
            tgt = 0.0
        else:
            w_star = min(target / s[t], cap) if s[t] > 0 else 0.0
            if w_prev == 0.0:
                tgt = w_star
            else:
                implied = w_prev * s[t]
                if implied < band[0] or implied > band[1]:
                    tgt = w_star
        cost = abs(tgt - w_prev) * COST
        w[t] = tgt
        ret[t] = tgt * rr[t] + (1 - tgt) * f[t] - cost
        gross = 1 + tgt * rr[t] + (1 - tgt) * f[t]
        w_prev = tgt * (1 + rr[t]) / gross if gross > 0 else 0.0
    return pd.Series(ret, index=r.index), pd.Series(w, index=r.index)


def stats(ret: pd.Series, w: pd.Series, rf: pd.Series, target: float) -> dict:
    yrs = len(ret) / 252
    roll = ret.rolling(63).std() * np.sqrt(252)
    return {"Sharpe": float(paris.sharpe(ret, rf=rf)), "Martin": float(paris.martin_ratio(ret)),
            "MaxDD": float(paris.max_drawdown(ret)), "CAGR": float(paris.cagr(ret)),
            "RealVol": float(paris.volatility(ret)), "|vol-target|": float(abs(paris.volatility(ret) - target)),
            "TrackRMSE63": float(np.sqrt(((roll.dropna() - target) ** 2).mean())),
            "Turnover/yr": float(w.diff().abs().sum() / yrs), "MeanW": float(w.mean())}


def panel(name: str, items: list[tuple[str, str]], target: float, band: tuple[float, float], cap: float,
          calibration: str, rf: pd.Series) -> pd.DataFrame:
    rows = []
    for sig_t, veh_t in items:
        sig_px, veh_px = load(sig_t), load(veh_t)
        r1 = sig_px["C"].pct_change().dropna()
        gate_trend = paris.trend_states(r1, jump_penalty=5.0, calibration=calibration)
        rv = veh_px["C"].pct_change().dropna()
        sigmas = {k: fn(veh_px).shift(1) for k, fn in ESTIMATORS.items()}  # known at close T-1
        start = max([s.dropna().index[0] for s in sigmas.values()] + [gate_trend.dropna().index[0], rv.index[0]])
        idx = rv.loc[start:].index
        rf_ = rf.reindex(idx).ffill().fillna(0.0)
        for est, sig in sigmas.items():
            sig = sig.reindex(idx)
            for gname, gate in (("none", pd.Series(1.0, index=idx)), ("trend", gate_trend.reindex(idx).ffill().fillna(0.0))):
                ret, w = run(gate, rv.loc[idx], rf_, sig, target, band, cap)
                rows.append({"panel": name, "signal": sig_t, "vehicle": veh_t, "estimator": est, "gate": gname,
                             "start": idx[0].date(), "days": len(idx), **stats(ret, w, rf_, target)})
        print(f"  {name} {veh_t}: {idx[0].date()} .. {idx[-1].date()}", flush=True)
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, est: str, col: str, rng, gate: str) -> tuple[float, float, float, int, int]:
    sub = df[df.gate == gate].pivot(index="vehicle", columns="estimator", values=col)
    d = (sub[est] - sub["E0 EWMA c2c"]).dropna()
    boot = [np.median(rng.choice(d.values, len(d))) for _ in range(1000)]
    return float(d.median()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int((d < 0).sum()), len(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="rolling", choices=("rolling", "expanding"))
    args = ap.parse_args()
    rf = load_rf()
    p1 = PREREG["panels"]["P1_1x"]
    p2 = PREREG["panels"]["P2_3x"]
    res = pd.concat([
        panel("P1 1x", [(t, t) for t in TICKERS], p1["sigma_target"], tuple(p1["band"]), p1["cap"], args.calibration, rf),
        panel("P2 3x", list(PAIRS.items()), p2["sigma_target"], tuple(p2["band"]), p2["cap"], args.calibration, rf),
    ], ignore_index=True)
    res.to_csv(OUT / "all_metrics.csv", index=False)
    rng = np.random.default_rng(0)
    L = [f"# Volatility estimators inside the vol-target loop — {pd.Timestamp.today().date()}\n",
         f"Pre-registration: `preregistration.json`. Gate = trend λ5 on the 1x, calibration **{args.calibration}**. "
         "5+3 bp one-way, T-bill cash, sigma and gate at close T−1. Tracking metrics on the ungated loop; outcome metrics on the gated loop.\n"]
    verdict = {}
    for pname, tgt, nmin in (("P1 1x", p1["sigma_target"], 6), ("P2 3x", p2["sigma_target"], 5)):
        sub = res[res.panel == pname]
        L.append(f"\n## {pname} (target {tgt:.0%}; cross-vehicle medians)\n")
        L.append("| Estimator | Gate | RealVol | Track RMSE63 | Sharpe | Martin | MaxDD | CAGR | Turnover/yr | Mean w |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        med = sub.groupby(["estimator", "gate"])[["RealVol", "TrackRMSE63", "Sharpe", "Martin", "MaxDD", "CAGR", "Turnover/yr", "MeanW"]].median()
        for est in ESTIMATORS:
            for g in ("none", "trend"):
                m = med.loc[(est, g)]
                L.append(f"| {est} | {g} | {m.RealVol:.1%} | {m.TrackRMSE63:.1%} | {m.Sharpe:.2f} | {m.Martin:.2f} | {m.MaxDD:.1%} | {m.CAGR:.1%} | {m['Turnover/yr']:.1f} | {m.MeanW:.2f} |")
        L.append(f"\n### Decision rule, {pname} (paired vs E0)\n")
        for est in list(ESTIMATORS)[1:]:
            dt, tlo, thi, twins, n = paired(sub, est, "TrackRMSE63", rng, "none")
            ds, slo, shi, _, _ = paired(sub, est, "Sharpe", rng, "trend")
            ds = -ds; slo, shi = -shi, -slo  # paired() counts "lower"; flip sign for Sharpe (higher better)
            dm, _, _, _, _ = paired(sub, est, "Martin", rng, "trend")
            turn = sub[sub.gate == "trend"].pivot(index="vehicle", columns="estimator", values="Turnover/yr")
            turn_ratio = float((turn[est] / turn["E0 EWMA c2c"]).median())
            adopt = twins >= nmin and ds >= -0.02 and turn_ratio <= 1.25
            v = "ADOPT" if adopt else ("RETIRE — measurement better, sizing worse" if twins >= nmin else "RETIRE")
            verdict[f"{pname} {est}"] = v
            L.append(f"**{est}: {v}** — ΔTrackRMSE {dt:+.2%} (CI [{tlo:+.2%}, {thi:+.2%}]), better {twins}/{n}; ΔSharpe (gated) {ds:+.2f} (CI [{slo:+.2f}, {shi:+.2f}]); ΔMartin {-dm:+.2f}; turnover ×{turn_ratio:.2f}.\n")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")  # tables only; readings live in READING.md
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
