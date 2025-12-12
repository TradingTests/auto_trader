import csv
import os
from datetime import datetime, timedelta, timezone
import pandas as pd


FIELDS = ["ts", "quote", "src", "symbol", "latest", "best_buy", "best_sell", "volume_src", "day_change"]


def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def append_snapshots(stats: dict, quote_currency: str, allowed_src: set, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)

    ts = datetime.now(timezone.utc).isoformat()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()

        for market, s in stats.items():
            # market like "btc-usdt"
            if "-" not in market:
                continue
            src, dst = market.split("-", 1)
            src = src.lower()
            dst = dst.lower()
            if dst != quote_currency.lower():
                continue
            if allowed_src and src not in allowed_src:
                continue
            if s.get("isClosed") is True:
                continue

            w.writerow(
                {
                    "ts": ts,
                    "quote": dst,
                    "src": src,
                    "symbol": f"{src}-{dst}",
                    "latest": _to_float(s.get("latest")),
                    "best_buy": _to_float(s.get("bestBuy")),
                    "best_sell": _to_float(s.get("bestSell")),
                    "volume_src": _to_float(s.get("volumeSrc")),
                    "day_change": _to_float(s.get("dayChange")),
                }
            )


def load_recent_history(csv_path: str, hours: int, allowed_src: set, quote_currency: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=FIELDS)

    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "latest"])
    df["src"] = df["src"].astype(str).str.lower()
    df["quote"] = df["quote"].astype(str).str.lower()

    df = df[df["quote"] == quote_currency.lower()]
    if allowed_src:
        df = df[df["src"].isin(list(allowed_src))]

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    df = df[df["ts"] >= cutoff].sort_values(["ts", "src"])
    return df.reset_index(drop=True)
