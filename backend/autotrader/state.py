import os
import json
from typing import Dict, Any
import pandas as pd

STATE_FILE = "backend/data/portfolio_state.json"


def _default_portfolio(starting_cash: float) -> Dict[str, Any]:
    return {
        "cash": float(starting_cash),
        "holdings": {},            # coin -> qty
        "equity_history": [],      # [{ts,value,cash}]
        "total_trades": 0,
        "tax_paid": 0.0,
    }


def load_state(starting_cash: float) -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            if isinstance(s, dict):
                # minimal migration safety
                s.setdefault("version", 1)
                s.setdefault("starting_cash", float(starting_cash))
                s.setdefault("selected_algorithm", "")
                s.setdefault("main", _default_portfolio(starting_cash))
                s.setdefault("algos", {})
                return s
        except Exception:
            pass

    return {
        "version": 1,
        "starting_cash": float(starting_cash),
        "selected_algorithm": "",
        "main": _default_portfolio(starting_cash),
        "algos": {},
    }


def save_state(state: Dict[str, Any], max_history_points: int = 4000):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    # trim histories to keep repo small
    def trim_portfolio(p: Dict[str, Any]):
        eh = p.get("equity_history", [])
        if isinstance(eh, list) and len(eh) > max_history_points:
            p["equity_history"] = eh[-max_history_points:]

    trim_portfolio(state.get("main", {}))
    for _, p in (state.get("algos", {}) or {}).items():
        if isinstance(p, dict):
            trim_portfolio(p)

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def portfolio_value(portfolio: Dict[str, Any], prices: Dict[str, float]) -> float:
    v = float(portfolio.get("cash", 0.0))
    holdings = portfolio.get("holdings", {}) or {}
    for coin, qty in holdings.items():
        px = float(prices.get(coin, 0.0) or 0.0)
        if px > 0 and qty:
            v += float(qty) * px
    return float(v)


def trim_equity_history_by_hours(portfolio: Dict[str, Any], hours: int):
    eh = portfolio.get("equity_history", [])
    if not eh or hours <= 0:
        return
    try:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
        kept = []
        for p in eh:
            ts = pd.to_datetime(p.get("ts"), utc=True, errors="coerce")
            if ts is not pd.NaT and ts >= cutoff:
                kept.append(p)
        portfolio["equity_history"] = kept
    except Exception:
        # if parsing fails, do nothing
        return
