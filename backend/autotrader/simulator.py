from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from .config import Config
from .strategies import compute_all_scores


@dataclass
class SimulationResult:
    generated_at: str
    period: dict
    summary: dict
    equity_curve: List[dict]
    trades: List[dict]
    latest_scores: List[dict]
    latest_allocations: Dict[str, float]
    debug: Dict[str, Any]


def _now_iso():
    return pd.Timestamp.utcnow().isoformat()


def _portfolio_value(cash: float, holdings: Dict[str, float], prices: Dict[str, float]) -> float:
    v = cash
    for coin, qty in holdings.items():
        px = prices.get(coin)
        if px is None or not np.isfinite(px) or px <= 0:
            continue
        v += qty * px
    return float(v)


def _get_prices_at(df_ts: pd.DataFrame) -> Dict[str, float]:
    # df for one timestamp, rows per src
    out = {}
    for _, r in df_ts.iterrows():
        out[str(r["src"]).lower()] = float(r["latest"])
    return out


def _select_targets(scores: Dict[str, float], max_positions: int, min_score: float) -> Dict[str, float]:
    # returns weights per coin, sums to <=1 (rest is cash)
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    chosen = [(c, s) for c, s in items if s >= min_score][:max_positions]
    if not chosen:
        return {}
    w = 1.0 / len(chosen)
    return {c: w for c, _ in chosen}


def run_simulation(cfg: Config, history: pd.DataFrame) -> SimulationResult:
    if history.empty:
        return SimulationResult(
            generated_at=_now_iso(),
            period={"start": None, "end": None, "hours": cfg.simulation_hours},
            summary={"note": "no_history_yet"},
            equity_curve=[],
            trades=[],
            latest_scores=[],
            latest_allocations={},
            debug={},
        )

    # Ensure evenly sorted
    history = history.sort_values(["ts", "src"]).reset_index(drop=True)
    ts_list = history["ts"].drop_duplicates().tolist()

    cash = float(cfg.starting_cash)
    holdings: Dict[str, float] = {}
    equity_curve = []
    trades: List[dict] = []
    tax_paid = 0.0

    # rebalance each snapshot (5min) by default; you can change to coarser if desired
    for i, ts in enumerate(ts_list):
        df_ts = history[history["ts"] == ts]
        prices = _get_prices_at(df_ts)

        # compute scores using data up to this timestamp
        panel = history[history["ts"] <= ts]
        scores, score_details = compute_all_scores(panel, cfg.strategy_weights)

        target_w = _select_targets(scores, cfg.max_positions, cfg.min_score_to_invest)

        value = _portfolio_value(cash, holdings, prices)

        # current weights
        current_w = {}
        for c, qty in holdings.items():
            px = prices.get(c)
            if px and px > 0:
                current_w[c] = (qty * px) / value

        # Rebalance: sell what is not in target, then buy target
        # 1) Sell
        for c in list(holdings.keys()):
            if c not in target_w:
                px = prices.get(c)
                if not px or px <= 0:
                    continue
                qty = holdings[c]
                notional = qty * px
                # slippage on sell => worse price
                exec_px = px * (1.0 - cfg.slippage_rate)
                proceeds = qty * exec_px
                tax = proceeds * cfg.tax_rate
                cash += (proceeds - tax)
                tax_paid += tax
                trades.append(
                    {
                        "ts": str(ts),
                        "type": "SELL",
                        "coin": c,
                        "qty": qty,
                        "price": exec_px,
                        "notional": proceeds,
                        "tax": tax,
                        "reason": "not_in_target",
                    }
                )
                del holdings[c]

        # recompute value after sells
        value = _portfolio_value(cash, holdings, prices)

        # 2) Buy / adjust to target weights
        # Convert target weights to target quantities
        targets_qty = {}
        for c, w in target_w.items():
            px = prices.get(c)
            if not px or px <= 0:
                continue
            target_value = value * w
            targets_qty[c] = target_value / px

        # Execute buys/adjustments (only buys in this simplified approach after selling non-targets)
        for c, tgt_qty in targets_qty.items():
            px = prices.get(c)
            if not px or px <= 0:
                continue
            cur_qty = holdings.get(c, 0.0)
            delta = tgt_qty - cur_qty
            if delta <= 1e-12:
                holdings[c] = cur_qty
                continue

            # buy with slippage
            exec_px = px * (1.0 + cfg.slippage_rate)
            cost = delta * exec_px
            tax = cost * cfg.tax_rate
            total_cost = cost + tax

            # if not enough cash, scale down
            if total_cost > cash and cash > 0:
                scale = cash / total_cost
                delta *= scale
                cost = delta * exec_px
                tax = cost * cfg.tax_rate
                total_cost = cost + tax

            if delta <= 1e-12:
                continue

            cash -= total_cost
            tax_paid += tax
            holdings[c] = cur_qty + delta
            trades.append(
                {
                    "ts": str(ts),
                    "type": "BUY",
                    "coin": c,
                    "qty": delta,
                    "price": exec_px,
                    "notional": cost,
                    "tax": tax,
                    "reason": "rebalance_to_target",
                }
            )

        # mark-to-market
        value = _portfolio_value(cash, holdings, prices)
        equity_curve.append({"ts": str(ts), "value": value, "cash": cash})

    start_ts = str(ts_list[0]) if ts_list else None
    end_ts = str(ts_list[-1]) if ts_list else None
    ending_value = equity_curve[-1]["value"] if equity_curve else cfg.starting_cash
    pnl = ending_value - cfg.starting_cash
    pnl_pct = (pnl / cfg.starting_cash) * 100.0 if cfg.starting_cash else 0.0

    # latest scores (for UI)
    final_panel = history
    scores, details = compute_all_scores(final_panel, cfg.strategy_weights)
    latest_scores = []
    for c, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        latest_scores.append({"coin": c, "score": sc, "details": details.get(c, {})})

    # latest allocations
    latest_prices = _get_prices_at(history[history["ts"] == ts_list[-1]])
    latest_value = _portfolio_value(cash, holdings, latest_prices)
    alloc = {"cash": cash / latest_value if latest_value else 1.0}
    for c, qty in holdings.items():
        px = latest_prices.get(c)
        if px and px > 0:
            alloc[c] = (qty * px) / latest_value

    return SimulationResult(
        generated_at=_now_iso(),
        period={"start": start_ts, "end": end_ts, "hours": cfg.simulation_hours},
        summary={
            "starting_cash": cfg.starting_cash,
            "ending_value": ending_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "num_trades": len(trades),
            "tax_paid": tax_paid,
            "tax_percent": cfg.tax_percent,
            "slippage_bps": cfg.slippage_bps,
        },
        equity_curve=equity_curve,
        trades=trades[-500:],  # cap for file size
        latest_scores=latest_scores,
        latest_allocations=alloc,
        debug={"holdings_qty": holdings},
    )
