"""Cross-asset summary of a walk-forward table folder: the CROSS_ASSET.md numbers.

For each metric, each rule's value per series, its mean over the eight ETFs (MKT excluded) and
the number of ETFs on which it beats buy-and-hold (higher is better for every metric printed,
MaxDD included, since drawdowns are negative). Run on the recorded tables it reproduces the
numbers in CROSS_ASSET.md; run on a re-run folder it gives the same summary for that run.

Usage:
    python research/walkforward_2026-09-02/summarize.py [TABLE_DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SERIES = ("MKT", "SPY", "QQQ", "XLK", "IWM", "EFA", "EEM", "TLT", "GLD")  # CROSS_ASSET order
ETFS = SERIES[1:]
RULES = {  # CROSS_ASSET.md label -> table row
    "B&H": "Buy & hold",
    "risk λ50": "fixed risk λ=50",
    "trend λ5": "fixed trend λ=5",
    "graded": "fixed graded r50/t5",
    "gate": "fixed gate r50/t5",
    "and": "fixed and r50/t5",
    "or": "fixed or r50/t5",
    "cells (causal sizing)": "cells r50/t5 (annual sizing)",
    "joint (logvol,slow) K=2 (causal)": "joint2f K=2 λ=5 (annual sizing)",
    "joint (logvol,slow,fast) K=4 (causal)": "joint3f K=4 λ=5 (annual sizing)",
    "selected annually (causal)": "selected annually (all candidates)",
}
METRICS = ("Sharpe", "MaxDD", "Martin", "CAGR", "Turnover/yr")


def summary(table_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {s: pd.read_csv(table_dir / f"{s}_table.csv", index_col=0) for s in SERIES}
    out = {}
    for m in METRICS:
        v = pd.DataFrame({s: tables[s].loc[list(RULES.values()), m] for s in SERIES})
        v.index = list(RULES)
        v["mean (ETFs)"] = v[list(ETFS)].mean(axis=1)
        v["wins vs B&H (of 8)"] = (v[list(ETFS)].gt(v.loc["B&H", list(ETFS)], axis=1)
                                   .sum(axis=1))
        out[m] = v
    return out


def main(dirs: list[str]) -> None:
    for d in dirs or [str(HERE)]:
        print(f"# {d}")
        for m, v in summary(Path(d)).items():
            nd = 2 if m == "Turnover/yr" else 3
            print(f"\n## {m}\n")
            print(v.round(nd).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
