"""Fetch dividend-adjusted daily OHLC closes from FMP (premium) into research/data_cache/.

Not shipped data; the cache directory is git-ignored. Chunks by ~4 calendar years because the
endpoint caps a single response at 5,000 rows.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import keyring
import pandas as pd
import requests

KEY = keyring.get_password("fmp", "api_key") or keyring.get_password("fmp_api", "api_key")
BASE = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
TSY = "https://financialmodelingprep.com/stable/treasury-rates"
OUT = Path(__file__).resolve().parent / "data_cache"


def fetch(symbol: str, start: str = "1990-01-01") -> pd.DataFrame:
    frames = []
    years = list(range(int(start[:4]), pd.Timestamp.today().year + 1, 4))
    for i, y in enumerate(years):
        frm = f"{y}-01-01"
        to = f"{years[i + 1] - 1}-12-31" if i + 1 < len(years) else pd.Timestamp.today().strftime("%Y-%m-%d")
        r = requests.get(BASE, params={"symbol": symbol, "from": frm, "to": to, "apikey": KEY}, timeout=60)
        r.raise_for_status()
        js = r.json()
        if js:
            frames.append(pd.DataFrame(js))
        time.sleep(0.2)
    if not frames:
        raise SystemExit(f"{symbol}: no data returned")
    df = pd.concat(frames).drop_duplicates("date").assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date").sort_index()
    if len(df) >= 5000 and any(len(f) == 5000 for f in frames):
        raise SystemExit(f"{symbol}: a chunk hit the 5,000-row cap — shrink the chunk size")
    return df


def fetch_treasury(start: str = "1990-01-01") -> pd.DataFrame:
    frames = []
    years = list(range(int(start[:4]), pd.Timestamp.today().year + 1, 4))
    for i, y in enumerate(years):
        to = f"{years[i + 1] - 1}-12-31" if i + 1 < len(years) else pd.Timestamp.today().strftime("%Y-%m-%d")
        r = requests.get(TSY, params={"from": f"{y}-01-01", "to": to, "apikey": KEY}, timeout=60)
        r.raise_for_status()
        if r.json():
            frames.append(pd.DataFrame(r.json()))
        time.sleep(0.2)
    df = pd.concat(frames).drop_duplicates("date").assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date").sort_index()
    return df


def main(symbols: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    if "TREASURY" in symbols:
        symbols.remove("TREASURY")
        t = fetch_treasury()
        t.to_csv(OUT / "treasury_rates.csv")
        print(f"treasury: {t.index[0].date()} .. {t.index[-1].date()} ({len(t)} rows)", flush=True)
    for s in symbols:
        df = fetch(s)
        cols = [c for c in ("adjOpen", "adjHigh", "adjLow", "adjClose", "volume") if c in df.columns]
        df[cols].to_csv(OUT / f"{s}_adj.csv")
        print(f"{s}: {df.index[0].date()} .. {df.index[-1].date()} ({len(df)} rows) cols={cols}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["SPY", "QQQ", "XLK", "IWM", "EFA", "EEM", "TLT", "GLD"])
