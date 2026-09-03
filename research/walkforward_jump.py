"""Walk-forward evaluation of the jump-model indicators and their combinations.

Protocol (agreed 2026-09-02): expanding-window selection with annual re-selection. For each
calendar year Y, every candidate exposure rule is scored on its own online-inferred history up to
the last day of Y-1 (net of a one-way cost on exposure changes); the best candidate is applied
during Y. The candidate states themselves come from the library's rolling 1,260-day / monthly-refit
/ lag-1 machinery, so nothing after T-1 enters the exposure for T at any stage. Rows labelled
"fixed" apply one rule with the library defaults throughout, for reference.

Usage:
    python research/walkforward_jump.py --ff  <F-F_Research_Data_Factors_daily.CSV>
                                       --etf <adjclose.csv with one column per ticker>
                                       --out research/walkforward_2026-09-02
Data are not shipped; the Fama-French file is public (Kenneth French Data Library) and the ETF
file is any dividend-adjusted daily close table (FMP was used).
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paris  # noqa: E402

RISK_GRID = (10.0, 20.0, 50.0, 100.0)
TREND_GRID = (2.0, 5.0, 10.0, 20.0)
JOINT_GRID = (5.0, 20.0)
COST = 0.0010  # one-way, on |change in exposure|
FIRST_SELECTION_YEARS = 3  # years of online history before the first annual selection


def load_ff(path: str) -> pd.DataFrame:
    lines = open(path, encoding="latin-1").read().splitlines()
    rows = [line for line in lines if re.match(r"^\s*\d{8}\s*,", line)]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=["date", "MktRF", "SMB", "HML", "RF"])
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date") / 100.0
    return pd.DataFrame({"MKT": df["MktRF"] + df["RF"], "RF": df["RF"]})


def net_returns(exposure: pd.Series, r: pd.Series, rf: pd.Series) -> pd.Series:
    e = exposure.fillna(0.0)
    cost = e.diff().abs().fillna(0.0) * COST
    return e * r + (1 - e) * rf - cost


def metrics(strat: pd.Series, exposure: pd.Series, rf: pd.Series, start: pd.Timestamp) -> dict:
    yrs = len(strat) / 252
    return {
        "CAGR": paris.cagr(strat, method="calendar", start=start - pd.Timedelta(days=1)),
        "Vol": paris.volatility(strat),
        "Sharpe": paris.sharpe(strat, rf=rf),
        "MaxDD": paris.max_drawdown(strat),
        "Ulcer%": paris.ulcer_index(strat, pct=True),
        "Martin": paris.martin_ratio(strat),
        "Time in": float(exposure.mean()),
        "Turnover/yr": float(exposure.diff().abs().sum() / yrs),
    }


def sizing_from_history(states: pd.Series, r: pd.Series, rf: pd.Series, upto: pd.Timestamp) -> dict:
    """Exposure per state value: clip(SR_k / max_k SR_k, 0, 1) on the history up to ``upto``."""
    s = states.loc[:upto].dropna()
    sr = {}
    for v in sorted(s.unique()):
        m = s == v
        if m.sum() <= 20:
            sr[v] = 0.0
            continue
        # per-state Sharpe from the library (validated against empyrical), per-period, rf-adjusted
        val = paris.sharpe(r.reindex(s.index)[m], rf=rf.reindex(s.index)[m], periods_per_year=252, annualize=False)
        sr[v] = float(val) if np.isfinite(val) else 0.0
    top = max(sr.values()) if sr else 0.0
    return {v: (float(np.clip(q / top, 0.0, 1.0)) if top > 0 else 0.0) for v, q in sr.items()}


def candidates(r: pd.Series, log) -> dict[str, dict]:
    """Every candidate: a dict with either an 'exposure' Series or 'states' + 'sized': True."""
    out: dict[str, dict] = {}
    risk = {lam: paris.risk_states(r, jump_penalty=lam) for lam in RISK_GRID}
    trend = {lam: paris.trend_states(r, jump_penalty=lam) for lam in TREND_GRID}
    log(f"  binary states done ({len(risk) + len(trend)} models)")
    for lr, R in risk.items():
        out[f"risk λ={lr:g}"] = {"exposure": R}
    for lt, T in trend.items():
        out[f"trend λ={lt:g}"] = {"exposure": T}
    for lr, R in risk.items():
        for lt, T in trend.items():
            for m in ("graded", "gate", "and", "or"):
                out[f"{m} r{lr:g}/t{lt:g}"] = {"exposure": paris.combine_states(R, T, m)}
            code = (R * 2 + T).where(R.notna() & T.notna())
            out[f"cells r{lr:g}/t{lt:g}"] = {"states": code, "sized": True}
    for feats, tag in ((("logvol", "slow"), "joint2f"), (("logvol", "slow", "fast"), "joint3f")):
        for k in (2, 4):
            for lam in JOINT_GRID:
                st = paris.joint_states(r, features=feats, n_states=k, jump_penalty=lam)
                out[f"{tag} K={k} λ={lam:g}"] = {"states": st, "sized": True}
    log(f"  joint models done; {len(out)} candidates")
    return out


def exposure_of(cand: dict, r: pd.Series, rf: pd.Series, upto: pd.Timestamp) -> pd.Series:
    """Exposure series of a candidate; state-sized candidates are sized on history up to ``upto``
    only (there is deliberately no whole-history option)."""
    if "exposure" in cand:
        return cand["exposure"]
    st = cand["states"]
    mp = sizing_from_history(st, r, rf, upto)
    return st.map(mp).where(st.notna())


def walk_forward(name: str, r: pd.Series, rf: pd.Series, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf = rf.reindex(r.index).fillna(0.0)
    cands = candidates(r, log)
    valid = next(iter(cands.values()))["exposure"] if "exposure" in next(iter(cands.values())) else None
    first = max(c["exposure"].first_valid_index() if "exposure" in c else c["states"].first_valid_index()
                for c in cands.values())
    years = sorted(set(r.loc[first:].index.year))
    start_year = years[0] + FIRST_SELECTION_YEARS
    oos = r.loc[str(start_year):]
    if len(oos) < 252:
        raise SystemExit(f"{name}: not enough history for a walk-forward")
    log(f"  OOS {oos.index[0].date()} .. {oos.index[-1].date()} ({len(oos)} days)")
    # annual selection
    selected = pd.Series(np.nan, index=oos.index)
    picks = []
    for y in range(start_year, years[-1] + 1):
        upto = pd.Timestamp(f"{y - 1}-12-31")
        best, best_sr = None, -np.inf
        for key, c in cands.items():
            e = exposure_of(c, r, rf, upto).loc[first:upto]
            strat = net_returns(e, r.loc[e.index], rf.loc[e.index])
            sr = paris.sharpe(strat, rf=rf.loc[e.index], periods_per_year=252)  # library Sharpe (empyrical-validated)
            sr = float(sr) if np.isfinite(sr) else -np.inf
            if sr > best_sr:
                best, best_sr = key, sr
        e_y = exposure_of(cands[best], r, rf, upto).loc[str(y)]
        selected.loc[e_y.index] = e_y
        picks.append({"year": y, "selected": best, "selection Sharpe": round(best_sr, 3)})
    rows = {}
    s0 = oos.index[0]
    bh = pd.Series(1.0, index=oos.index)
    rows["Buy & hold"] = metrics(net_returns(bh, oos, rf.loc[oos.index]), bh, rf.loc[oos.index], s0)
    fixed = {
        "fixed risk λ=50": cands["risk λ=50"], "fixed trend λ=5": cands["trend λ=5"],
        "fixed graded r50/t5": cands["graded r50/t5"], "fixed gate r50/t5": cands["gate r50/t5"],
        "fixed and r50/t5": cands["and r50/t5"], "fixed or r50/t5": cands["or r50/t5"],
    }
    for label, c in fixed.items():
        e = c["exposure"].loc[oos.index]
        rows[label] = metrics(net_returns(e, oos, rf.loc[oos.index]), e, rf.loc[oos.index], s0)
    # causal state-conditional sizing with default penalties (re-sized annually on the expanding history)
    for label, key in (("cells r50/t5 (annual sizing)", "cells r50/t5"),
                       ("joint2f K=2 λ=5 (annual sizing)", "joint2f K=2 λ=5"),
                       ("joint3f K=4 λ=5 (annual sizing)", "joint3f K=4 λ=5")):
        e = pd.Series(np.nan, index=oos.index)
        for y in range(start_year, years[-1] + 1):
            ey = exposure_of(cands[key], r, rf, pd.Timestamp(f"{y - 1}-12-31")).loc[str(y)]
            e.loc[ey.index] = ey
        e = e.fillna(0.0)
        rows[label] = metrics(net_returns(e, oos, rf.loc[oos.index]), e, rf.loc[oos.index], s0)
    sel = selected.fillna(0.0)
    rows["selected annually (all candidates)"] = metrics(net_returns(sel, oos, rf.loc[oos.index]), sel, rf.loc[oos.index], s0)
    table = pd.DataFrame(rows).T
    table.index.name = name
    return table, pd.DataFrame(picks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff", required=True)
    ap.add_argument("--etf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ff = load_ff(args.ff)
    px = pd.read_csv(args.etf, index_col=0, parse_dates=True)
    series = {"MKT (Fama-French)": ff["MKT"].loc["1987":]}
    for c in px.columns:
        series[c] = px[c].dropna().pct_change().dropna()
    rf = ff["RF"]
    md = [f"# Walk-forward evaluation of jump-model indicators — {pd.Timestamp.today().date()}\n",
          "Protocol: expanding-window selection, annual re-selection, 10 bp one-way cost, cash = T-bill. "
          "States: rolling 1,260-day calibration, monthly refit, lag 1. 'fixed' rows use library defaults "
          "throughout; '(annual sizing)' rows map states to exposure with clip(SR_k / max SR, 0, 1) estimated "
          "on the expanding history; 'selected annually' picks the best of all candidates each year.\n"]
    for name, r in series.items():
        t0 = time.time()
        print(f"{name}: {r.index[0].date()} .. {r.index[-1].date()}", flush=True)
        table, picks = walk_forward(name, r, rf, lambda m: print(m, flush=True))
        table.to_csv(out / f"{name.split()[0]}_table.csv")
        picks.to_csv(out / f"{name.split()[0]}_picks.csv", index=False)
        try:
            body = table.round(3).to_markdown()
        except ImportError:  # tabulate not installed
            body = "```\n" + table.round(3).to_string() + "\n```"
        md.append(f"\n## {name}\n\n" + body + "\n\nAnnual picks: "
                  + ", ".join(f"{p.year}: {p.selected}" for p in picks.itertuples()) + "\n")
        print(table.round(3).to_string(), flush=True)
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
    (out / "WALKFORWARD.md").write_text("\n".join(md))
    print("written", out / "WALKFORWARD.md")


if __name__ == "__main__":
    main()
