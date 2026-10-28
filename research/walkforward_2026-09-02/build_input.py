"""Rebuild the input of the 2026-09-02 walk-forward run, cut at its recorded end date.

The 2026-09-02 run read one dividend-adjusted daily close table (one column per ticker) that was
never recorded. This script rebuilds it from the research data cache written by
``research/fetch_fmp.py`` (FMP ``stable/historical-price-eod/dividend-adjusted``, column
``adjClose``, fetched 2026-09-03), cut at 2026-09-02, in the recorded column order, and checks it
against the SHA-256 of the table that reproduces the recorded tables byte for byte (verified
2026-10-28 with the 0.10.0 source; see ``REPRODUCE.md``). Vendor data are not shipped: the table
goes to the git-ignored ``input/`` folder.

Usage:
    python research/walkforward_2026-09-02/build_input.py [--cache DIR] [--ff FILE]
    python research/walkforward_jump.py --ff <F-F_Research_Data_Factors_daily.CSV> \
        --etf research/walkforward_2026-09-02/input/etf_adjclose.csv --out <a new folder>
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
END = "2026-09-02"  # last bar of the recorded run
TICKERS = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLK")  # recorded column order
INPUT_SHA256 = "19485b8f39df81c24bbc0fad0a5dc1879d2bf4a8c1c9c138bb27b3a4cf909921"
# Kenneth French Data Library, daily three factors, 202606 vintage (last row 20260630)
FF_SHA256 = "3d73942f7c0ba3accd021721876fef3059c6b02764a476dbdd9c311817f09710"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(cache: Path) -> pd.DataFrame:
    cols = {}
    for t in TICKERS:
        f = cache / f"{t}_adj.csv"
        if not f.exists():
            raise SystemExit(f"{f} is missing: run `python research/fetch_fmp.py "
                             f"{' '.join(TICKERS)}` first (see REPRODUCE.md)")
        cols[t] = pd.read_csv(f, index_col="date", parse_dates=True)["adjClose"].loc[:END]
    return pd.DataFrame(cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=HERE.parent / "data_cache")
    ap.add_argument("--out", type=Path, default=HERE / "input" / "etf_adjclose.csv")
    ap.add_argument("--ff", type=Path, help="also check the Fama-French daily file")
    args = ap.parse_args()
    px = build(args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(args.out)
    got = sha256(args.out)
    print(f"{args.out}: {px.index[0].date()} .. {px.index[-1].date()}, {len(px)} rows, "
          f"sha256 {got}")
    problems = []
    if got != INPUT_SHA256:
        problems.append(
            f"input sha256 {got} differs from the recorded {INPUT_SHA256}. A re-fetch after a "
            "later dividend rescales the adjusted history, so returns can differ in the last "
            "digits; the tables may then differ from the record")
    if args.ff is not None:
        ff = sha256(args.ff)
        print(f"{args.ff}: sha256 {ff}")
        if ff != FF_SHA256:
            problems.append(f"Fama-French sha256 {ff} differs from the recorded {FF_SHA256} "
                            "(202606 vintage); a later vintage moves the MKT table")
    if problems:
        raise SystemExit("\n".join(problems))
    print("inputs match the recorded run")


if __name__ == "__main__":
    main()
