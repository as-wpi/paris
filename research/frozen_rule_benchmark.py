"""PARIS jump-model regimes vs the frozen rtl-screener two-term rule (stage 1), and the amended-in
LLT signals as jump-model features (stage 2). Pre-registration: research/frozen_rule_benchmark/
preregistration.json — read it first; nothing here is tuned.

Runs in the rtl-workspace interpreter (numba, RTL-Kalman) with the PARIS source on sys.path:

    ~/.venvs/rtl-workspace/bin/python research/frozen_rule_benchmark.py --out research/frozen_rule_benchmark

The frozen rule is computed by the screener's own engine (RTLEKOscillators + rtl_snapshot._rolling_pr),
never re-derived; a parity assertion against build_rtl_snapshot on SPY guards that.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WORK = _HERE.parents[2]
for p in (_HERE.parent / "src",
          _WORK / "Quant Science" / "projects" / "rtl-screener" / "backend",
          _WORK / "Quant Science" / "projects" / "RTL-Kalman" / "src"):
    sys.path.insert(0, str(p))

import paris  # noqa: E402
from app.analytics.rtl_snapshot import (DEFAULT_RTL_CONFIG, _LLT_STRESS_PR, _rolling_pr,  # noqa: E402
                                        build_rtl_snapshot)
from app.analytics import gated_strategy as gs  # noqa: E402
from rtl_kalman.indicators.oscillators import RTLEKOscillators  # noqa: E402

DATA = _HERE / "data_cache"
FF = _WORK / "Personal_Trading_Strategies" / "Market Timing" / "shared_data" / "F-F_Research_Data_Factors_daily.CSV"
PREREG = json.loads((_HERE / "frozen_rule_benchmark" / "preregistration.json").read_text())
COST = PREREG["execution"]["cost_one_way"]
TICKERS = PREREG["universe"]
EXUS = ["EFA", "EEM", "TLT", "GLD"]
RISK_LAM, TREND_LAM = 50.0, 5.0


# ----------------------------------------------------------------------------- data
def load_rf() -> pd.Series:
    lines = open(FF, encoding="latin-1").read().splitlines()
    rows = [ln for ln in lines if re.match(r"^\s*\d{8}\s*,", ln)]
    ff = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=["date", "MktRF", "SMB", "HML", "RF"])
    ff["date"] = pd.to_datetime(ff["date"].astype(str), format="%Y%m%d")
    rf = ff.set_index("date")["RF"] / 100.0
    tsy = pd.read_csv(DATA / "treasury_rates.csv", index_col=0, parse_dates=True)["month3"]
    ext = (tsy / 100.0 / 252.0).shift(1)  # yield known at close T-1 earns on day T
    ext = ext.loc[ext.index > rf.index[-1]]
    return pd.concat([rf, ext]).sort_index()


def load_ohlc(t: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{t}_adj.csv", index_col=0, parse_dates=True)
    df = df.rename(columns={"adjOpen": "Open", "adjHigh": "High", "adjLow": "Low", "adjClose": "Close", "volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


# ----------------------------------------------------------------------------- screener features
def screener_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """x_raw, x_raw_llt, trend_strength, fit_quality, fq_pr (PR63), innov (InnovLMAR) at close T,
    from the screener's own engine with its frozen configuration."""
    df = ohlc.copy()
    df.columns = [c.lower() for c in df.columns]
    df["adj_close"] = df["close"]
    out = RTLEKOscillators(DEFAULT_RTL_CONFIG).compute_all(df)
    fq = out["fit_quality"].astype(float)
    return pd.DataFrame({"x_raw": out["x_raw"].astype(float), "x_raw_llt": out["x_raw_llt"].astype(float),
                         "ts": out["trend_strength"].astype(float), "fq": fq, "fq_pr": _rolling_pr(fq),
                         "innov": out["risk_innov_raw"].astype(float)}, index=ohlc.index)


def parity_check(ohlc: pd.DataFrame, feats: pd.DataFrame, log) -> None:
    snap = build_rtl_snapshot(ohlc, ticker="SPY")
    idx = pd.DatetimeIndex(pd.to_datetime(snap.dates_iso))
    sh = snap.shadow_llt
    for key, col in (("trend_strength_series", "ts"), ("fit_quality_pr_series", "fq_pr")):
        ref = pd.Series([np.nan if v is None else float(v) for v in sh[key]], index=idx)
        both = pd.concat([ref, feats[col].reindex(idx)], axis=1).dropna()
        diff = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
        log(f"  parity {col}: {len(both)} bars, max |diff| = {diff:.2e}")
        if diff > 1e-9:
            raise SystemExit(f"parity failure on {col}")


def frozen_risk_off(f: pd.DataFrame) -> pd.Series:
    return ((f["ts"] < 0) | (f["fq_pr"] > _LLT_STRESS_PR)).astype(bool)


# ----------------------------------------------------------------------------- evaluation
def net_returns(e: pd.Series, r: pd.Series, rf: pd.Series) -> pd.Series:
    e = e.reindex(r.index).fillna(0.0)
    cost = e.diff().abs().fillna(0.0) * COST
    return e * r + (1 - e) * rf.reindex(r.index) - cost


def metrics(e: pd.Series, r: pd.Series, rf: pd.Series) -> dict:
    s = net_returns(e, r, rf)
    e = e.reindex(r.index).fillna(0.0)
    yrs = len(s) / 252.0
    return {"Sharpe": float(paris.sharpe(s, rf=rf.reindex(s.index))), "MaxDD": float(paris.max_drawdown(s)),
            "Martin": float(paris.martin_ratio(s)), "Ulcer%": float(paris.ulcer_index(s, pct=True)),
            "CAGR": float(paris.cagr(s)), "TimeIn": float(e.mean()), "OffFrac": float(1.0 - e.mean()),
            "Flips/yr": float((e.diff().abs() > 1e-12).sum() / yrs)}


def folds() -> dict[str, tuple[pd.Timestamp, pd.Timestamp, str]]:
    out = {}
    for kind in ("bear_mandatory", "bull"):
        for k, (a, b) in PREREG["folds"][kind].items():
            out[k] = (pd.Timestamp(a), pd.Timestamp(b), kind)
    return out


def jump_on(features: pd.DataFrame, lam: float, high_is_on: bool) -> pd.Series:
    """Two-state jump model on a feature frame (first column orders the states); returns the
    on/off exposure for day T from the state inferred at close T-1."""
    s = paris.jump_states(features.dropna(), lam)
    on = s if high_is_on else 1.0 - s
    return on.shift(1)


def arms_for(t: str, ohlc: pd.DataFrame, feats: pd.DataFrame, spy_ro: pd.Series, r: pd.Series,
             stage2: bool, log) -> dict[str, pd.Series]:
    idx = r.index
    # frozen incumbent: features at close T-1 -> exposure for T (screener t+1 convention)
    f1 = feats.shift(1).reindex(idx)
    s1 = (~spy_ro.reindex(idx).ffill().fillna(False).astype(bool)).astype(float)  # SPY risk_off at T-1 (already shifted)
    s2 = gs.base_exposure(f1["x_raw"])
    s3 = gs.innov_risk_gated_exposure(pd.Series(1.0, index=idx), f1["innov"])
    stack = gs.stack_exposure(f1["x_raw"], ~s1.astype(bool), f1["innov"])
    # PARIS (states already lagged one day inside the library)
    trend = paris.trend_states(r, jump_penalty=TREND_LAM)
    risk = paris.risk_states(r, jump_penalty=RISK_LAM)
    arms = {
        "Buy & hold": pd.Series(1.0, index=idx),
        "Frozen S1 (SPY switch)": s1, "Frozen S2 (own x_raw>0)": s2, "Frozen S3 (own InnovLMAR<=1.25)": s3,
        "Frozen S1xS2": s1 * s2, "Frozen stack S1xS2xS3": stack,
        "PARIS trend λ5": trend, "PARIS risk λ50": risk,
        "PARIS gate": paris.combine_states(risk, trend, "gate"), "PARIS graded": paris.combine_states(risk, trend, "graded"),
        "PARIS trend λ5 x S1 (exploratory)": trend * s1,
    }
    if stage2:
        lv = paris.risk_signal(r, log=True)            # log EWMA vol at close T (level feature)
        slow = paris.trend_signal(r, "slow")           # slow trailing mean at close T (level feature)
        f0 = feats.reindex(idx)                        # screener features at close T
        F1 = jump_on(pd.DataFrame({"logvol": lv, "logfq": np.log(f0["fq"])}), RISK_LAM, high_is_on=False)
        F2 = jump_on(pd.DataFrame({"logvol": lv, "loginnov": np.log(f0["innov"])}), RISK_LAM, high_is_on=False)
        T1 = jump_on(pd.DataFrame({"slow": slow, "llt_drift": f0["x_raw_llt"]}), TREND_LAM, high_is_on=True)
        T2 = jump_on(pd.DataFrame({"slow": slow, "ts": f0["ts"]}), TREND_LAM, high_is_on=True)
        arms.update({
            "F1 risk+logFQ λ50": F1, "F1 gate (trend λ5 x F1)": paris.combine_states(F1, trend, "gate"),
            "F2 risk+logInnov λ50": F2, "F2 gate (trend λ5 x F2)": paris.combine_states(F2, trend, "gate"),
            "T1 trend+LLTdrift λ5": T1, "T2 trend+TS λ5": T2,
        })
    log(f"  {len(arms)} arms built")
    return arms


def evaluate(t: str, arms: dict[str, pd.Series], r: pd.Series, rf: pd.Series) -> pd.DataFrame:
    first = max(a.first_valid_index() for a in arms.values())
    start = max(first, r.index[0] + pd.Timedelta(days=1))
    rows = []
    fl = folds()
    for name, e in arms.items():
        e = e.loc[start:].reindex(r.loc[start:].index)
        m = metrics(e, r.loc[start:], rf)
        rows.append({"ticker": t, "arm": name, "fold": "FULL", "start": start.date(), "end": r.index[-1].date(), "days": len(e), **m})
        for fk, (a, b, kind) in fl.items():
            ef = e.loc[a:b]
            if len(ef) < 60 or ef.index[0] > a + pd.Timedelta(days=45):
                continue
            rows.append({"ticker": t, "arm": name, "fold": fk, "start": ef.index[0].date(), "end": ef.index[-1].date(),
                         "days": len(ef), **metrics(ef, r.loc[ef.index], rf)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- summary + decision
def summarise(res: pd.DataFrame, out: Path, stage2: bool) -> str:
    stack = "Frozen stack S1xS2xS3"
    fl = folds()
    lines = [f"# PARIS regimes vs the frozen rtl-screener rule — {pd.Timestamp.today().date()}\n",
             f"Pre-registration: `research/frozen_rule_benchmark/preregistration.json`. Universe {', '.join(TICKERS)}; "
             f"cost {COST*1e4:.0f} bp one-way; cash = T-bill; exposure for T from information ≤ close T−1.\n"]
    med = res.groupby(["fold", "arm"])[["Sharpe", "MaxDD", "Martin", "CAGR", "OffFrac", "Flips/yr"]].median()
    order = list(dict.fromkeys(res["arm"]))
    for fold in ["FULL"] + list(fl):
        if fold not in med.index.get_level_values(0):
            continue
        tb = med.loc[fold].reindex(order).dropna(how="all")
        n = res[res.fold == fold].groupby("arm").size().reindex(tb.index)
        kind = fl[fold][2] if fold in fl else "full OOS"
        lines.append(f"\n## {fold} ({kind}; cross-ticker medians, n = {int(n.max())} tickers)\n")
        lines.append("| Arm | Sharpe | MaxDD | Martin | CAGR | Off-frac | Flips/yr |\n|---|---:|---:|---:|---:|---:|---:|")
        for arm, row in tb.iterrows():
            lines.append(f"| {arm} | {row.Sharpe:.2f} | {row.MaxDD:.1%} | {row.Martin:.2f} | {row.CAGR:.1%} | {row.OffFrac:.2f} | {row['Flips/yr']:.1f} |")
    # wins vs stack, full OOS, per ticker
    full = res[res.fold == "FULL"].pivot(index="ticker", columns="arm", values="Sharpe")
    lines.append("\n## Full-OOS Sharpe by ticker\n")
    cols = [c for c in order if c in full.columns]
    lines.append("| Ticker | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols))
    for t, row in full.reindex(TICKERS).iterrows():
        lines.append(f"| {t} | " + " | ".join(f"{row[c]:.2f}" for c in cols) + " |")
    # decision rule
    lines.append("\n## Pre-registered decision rule\n")
    rng = np.random.default_rng(0)
    challengers = [a for a in order if a.startswith("PARIS") and "exploratory" not in a]
    if stage2:
        challengers += [a for a in order if a[:2] in ("F1", "F2", "T1", "T2")]
    verdicts = {}
    for ch in challengers:
        d_sh = (full[ch] - full[stack]).dropna()
        boot = [np.median(rng.choice(d_sh.values, len(d_sh))) for _ in range(1000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        d_mart = (res[res.fold == "FULL"].pivot(index="ticker", columns="arm", values="Martin")[ch]
                  - res[res.fold == "FULL"].pivot(index="ticker", columns="arm", values="Martin")[stack]).median()
        bear_ok, bull_ok, notes = True, True, []
        for fk, (a, b, kind) in fl.items():
            sub = res[res.fold == fk]
            if sub.empty:
                continue
            m = sub.groupby("arm")[["MaxDD", "OffFrac", "Sharpe"]].median()
            if ch not in m.index or stack not in m.index:
                continue
            if kind == "bear_mandatory":
                ok = m.loc[ch, "MaxDD"] >= m.loc[stack, "MaxDD"] - 0.01
                bear_ok &= bool(ok)
                notes.append(f"{fk}: MaxDD {m.loc[ch,'MaxDD']:.1%} vs stack {m.loc[stack,'MaxDD']:.1%} {'ok' if ok else 'FAIL'}; ΔSharpe {m.loc[ch,'Sharpe']-m.loc[stack,'Sharpe']:+.2f}")
            else:
                ok = (m.loc[ch, "OffFrac"] <= m.loc[stack, "OffFrac"] + 0.10) and (m.loc[ch, "OffFrac"] <= 0.35)
                bull_ok &= bool(ok)
                notes.append(f"{fk}: off-frac {m.loc[ch,'OffFrac']:.2f} vs stack {m.loc[stack,'OffFrac']:.2f} {'ok' if ok else 'FAIL'}")
        med_d = float(d_sh.median())
        if med_d >= -0.05 and bear_ok and bull_ok:
            v = "BEATS" if (med_d >= 0.10 and d_mart > 0) else "MATCHES"
        else:
            v = "LOSES"
        verdicts[ch] = v
        wins = int((d_sh > 0).sum())
        lines.append(f"\n**{ch} vs {stack}: {v}** — median ΔSharpe {med_d:+.2f} (95% CI [{lo:+.2f}, {hi:+.2f}]), wins {wins}/{len(d_sh)}, "
                     f"median ΔMartin {d_mart:+.2f}; bear folds {'pass' if bear_ok else 'FAIL'}; false-bear ceiling {'pass' if bull_ok else 'FAIL'}.")
        for nt in notes:
            lines.append(f"- {nt}")
    # attribution: US vs ex-US
    lines.append("\n## Attribution: US equity vs ex-US names (full-OOS median ΔSharpe vs stack)\n")
    lines.append("| Arm | US (SPY, QQQ, XLK, IWM) | ex-US (EFA, EEM, TLT, GLD) |\n|---|---:|---:|")
    for ch in challengers:
        d = full[ch] - full[stack]
        lines.append(f"| {ch} | {d.reindex([t for t in TICKERS if t not in EXUS]).median():+.2f} | {d.reindex(EXUS).median():+.2f} |")
    txt = "\n".join(lines) + "\n"
    (out / ("RESULTS.md" if stage2 else "STAGE1_RESULTS.md")).write_text(txt)
    return txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE / "frozen_rule_benchmark"))
    ap.add_argument("--stage2", action="store_true", help="also run the amended-in LLT feature tests")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    rf = load_rf()
    spy = load_ohlc("SPY")
    spy_f = screener_features(spy)
    parity_check(spy, spy_f, log)
    spy_ro = frozen_risk_off(spy_f)  # at close T; lagged and aligned per ticker below
    results = []
    for t in TICKERS:
        t0 = time.time()
        ohlc = spy if t == "SPY" else load_ohlc(t)
        feats = spy_f if t == "SPY" else screener_features(ohlc)
        r = ohlc["Close"].pct_change().dropna()
        log(f"{t}: {r.index[0].date()} .. {r.index[-1].date()} ({len(r)} days)")
        # SPY risk-off at close T-1 -> gate for day T, aligned to this ticker's calendar (causal ffill)
        ro_lag = spy_ro.shift(1).reindex(r.index, method="ffill")
        arms = arms_for(t, ohlc, feats, ro_lag, r, args.stage2, log)
        res = evaluate(t, arms, r, rf)
        res.to_csv(out / f"{t}_metrics.csv", index=False)
        pd.DataFrame(arms).to_csv(out / f"{t}_exposures.csv")
        results.append(res)
        log(f"  done in {time.time()-t0:.0f}s")
    res = pd.concat(results, ignore_index=True)
    res.to_csv(out / "all_metrics.csv", index=False)
    print(summarise(res, out, args.stage2))


if __name__ == "__main__":
    main()
