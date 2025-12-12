from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from .config import Config
from .strategies import compute_packs_by_algorithm
from .state import portfolio_value, trim_equity_history_by_hours


def _now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _latest_prices(history: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    if history.empty:
        return _now_iso(), {}
    last_ts = history["ts"].max()
    df = history[history["ts"] == last_ts]
    prices = {}
    for _, r in df.iterrows():
        c = str(r["src"]).lower()
        px = float(r["latest"]) if pd.notna(r["latest"]) else 0.0
        if px > 0:
            prices[c] = px
    return pd.Timestamp(last_ts).isoformat(), prices


def _select_targets(scores: Dict[str, float], max_positions: int, min_score: float, prices: Dict[str, float]) -> Dict[str, float]:
    # choose top coins by score; ignore missing prices
    items = [(c, float(s)) for c, s in (scores or {}).items() if float(prices.get(c, 0.0) or 0.0) > 0.0]
    items.sort(key=lambda x: x[1], reverse=True)

    chosen = [(c, s) for c, s in items if s >= float(min_score)][: int(max_positions)]
    if not chosen:
        return {}

    # rank-based weights (stable, always sums to 1)
    # rank1 gets k, rank2 gets k-1, ...
    k = len(chosen)
    raw = [k - i for i in range(k)]
    denom = float(sum(raw))
    return {chosen[i][0]: float(raw[i]) / denom for i in range(k)}


def _apply_trade_costs(notional: float, cfg: Config) -> Tuple[float, float]:
    # returns (tax, slippage_cost_placeholder) - we model slippage via exec price, so here only tax
    tax = float(notional) * cfg.tax_rate
    return tax, 0.0


def rebalance_portfolio(
    portfolio: Dict[str, Any],
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    cfg: Config,
    ts: str,
    min_trade_notional: float,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []

    holdings: Dict[str, float] = portfolio.get("holdings", {}) or {}
    cash = float(portfolio.get("cash", 0.0))

    # mark-to-market
    value = portfolio_value(portfolio, prices)
    if value <= 0:
        portfolio["cash"] = cash
        portfolio["holdings"] = holdings
        return actions

    # current weights
    current_w = {"cash": cash / value}
    for c, qty in list(holdings.items()):
        px = float(prices.get(c, 0.0) or 0.0)
        if px > 0 and qty > 0:
            current_w[c] = (qty * px) / value
        else:
            # drop invalid holding if no price
            holdings.pop(c, None)

    # SELL step: sell coins not in target or overweight vs target
    for c in list(holdings.keys()):
        px = float(prices.get(c, 0.0) or 0.0)
        if px <= 0:
            continue

        cw = float(current_w.get(c, 0.0))
        tw = float(target_weights.get(c, 0.0))

        if tw < cw:
            target_value = tw * value
            current_value = holdings[c] * px
            sell_value = current_value - target_value

            if sell_value < min_trade_notional:
                continue

            sell_qty = sell_value / px
            sell_qty = min(sell_qty, holdings[c])

            # slippage: worse price on sells
            exec_px = px * (1.0 - cfg.slippage_rate)
            proceeds = sell_qty * exec_px
            tax, _ = _apply_trade_costs(proceeds, cfg)
            net = proceeds - tax

            cash += net
            holdings[c] -= sell_qty
            if holdings[c] <= 1e-12:
                holdings.pop(c, None)

            portfolio["total_trades"] = int(portfolio.get("total_trades", 0)) + 1
            portfolio["tax_paid"] = float(portfolio.get("tax_paid", 0.0)) + float(tax)

            actions.append({
                "ts": ts,
                "type": "SELL",
                "coin": c,
                "qty": float(sell_qty),
                "price": float(exec_px),
                "notional": float(proceeds),
                "tax": float(tax),
                "reason": "rebalance",
            })

    # update value after sells
    portfolio["cash"] = cash
    portfolio["holdings"] = holdings
    value = portfolio_value(portfolio, prices)
    cash = float(portfolio["cash"])

    # BUY step: buy coins underweight vs target
    for c, tw in (target_weights or {}).items():
        px = float(prices.get(c, 0.0) or 0.0)
        if px <= 0:
            continue

        current_qty = float(holdings.get(c, 0.0))
        current_value = current_qty * px
        target_value = float(tw) * value
        buy_value = target_value - current_value

        if buy_value < min_trade_notional:
            continue
        if cash < min_trade_notional:
            break

        # slippage: worse price on buys
        exec_px = px * (1.0 + cfg.slippage_rate)
        buy_qty = buy_value / exec_px
        cost = buy_qty * exec_px
        tax, _ = _apply_trade_costs(cost, cfg)
        total_cost = cost + tax

        if total_cost > cash:
            # scale down to available cash
            scale = (cash * 0.995) / total_cost
            if scale <= 0:
                continue
            buy_qty *= scale
            cost = buy_qty * exec_px
            tax, _ = _apply_trade_costs(cost, cfg)
            total_cost = cost + tax

        if cost < min_trade_notional or buy_qty <= 1e-12:
            continue

        cash -= total_cost
        holdings[c] = float(holdings.get(c, 0.0)) + float(buy_qty)

        portfolio["total_trades"] = int(portfolio.get("total_trades", 0)) + 1
        portfolio["tax_paid"] = float(portfolio.get("tax_paid", 0.0)) + float(tax)

        actions.append({
            "ts": ts,
            "type": "BUY",
            "coin": c,
            "qty": float(buy_qty),
            "price": float(exec_px),
            "notional": float(cost),
            "tax": float(tax),
            "reason": "rebalance",
        })

    portfolio["cash"] = float(cash)
    portfolio["holdings"] = holdings
    return actions


def trailing_return(equity_history: List[dict], lookback_points: int) -> float:
    if not equity_history or len(equity_history) < 2:
        return 0.0
    end = float(equity_history[-1].get("value", 0.0) or 0.0)
    start_idx = max(0, len(equity_history) - int(lookback_points))
    start = float(equity_history[start_idx].get("value", 0.0) or 0.0)
    if start <= 0:
        return 0.0
    return (end / start) - 1.0


def run_step(cfg: Config, history: pd.DataFrame, state: Dict[str, Any]) -> Dict[str, Any]:
    ts, prices = _latest_prices(history)

    # ensure state structure
    state.setdefault("main", {"cash": cfg.starting_cash, "holdings": {}, "equity_history": [], "total_trades": 0, "tax_paid": 0.0})
    state.setdefault("algos", {})
    state.setdefault("selected_algorithm", "")

    main = state["main"]
    algos: Dict[str, Any] = state["algos"]

    packs = compute_packs_by_algorithm(history)

    # init algo portfolios
    for algo_name in packs.keys():
        if algo_name not in algos:
            algos[algo_name] = {
                "cash": float(cfg.starting_cash),
                "holdings": {},
                "equity_history": [],
                "total_trades": 0,
                "tax_paid": 0.0,
            }

    # ---- Update each algorithm shadow portfolio ----
    leaderboard = []
    algo_targets: Dict[str, Dict[str, float]] = {}

    for algo_name, pack in packs.items():
        p = algos[algo_name]

        target = _select_targets(
            scores=pack.scores,
            max_positions=cfg.max_positions,
            min_score=cfg.algo_min_score_to_invest,
            prices=prices,
        )
        algo_targets[algo_name] = target

        _ = rebalance_portfolio(
            portfolio=p,
            target_weights=target,
            prices=prices,
            cfg=cfg,
            ts=ts,
            min_trade_notional=cfg.min_trade_notional,
        )

        v = portfolio_value(p, prices)
        p.setdefault("equity_history", [])
        p["equity_history"].append({"ts": ts, "value": round(v, 4)})

        # keep bounded history
        if len(p["equity_history"]) > cfg.max_history_points:
            p["equity_history"] = p["equity_history"][-cfg.max_history_points:]

        r = trailing_return(p["equity_history"], cfg.algo_rank_lookback_points)

        leaderboard.append({
            "algorithm": algo_name,
            "value": round(v, 4),
            "return_lookback_pct": round(r * 100.0, 3),
            "holdings": sorted(list((p.get("holdings") or {}).keys())),
            "trades": int(p.get("total_trades", 0)),
            "tax_paid": round(float(p.get("tax_paid", 0.0)), 4),
        })

    leaderboard.sort(key=lambda x: (x["return_lookback_pct"], x["value"]), reverse=True)
    best_algo = leaderboard[0]["algorithm"] if leaderboard else state.get("selected_algorithm", "")
    state["selected_algorithm"] = best_algo

    # ---- Main portfolio follows the best algorithm (the "race winner") ----
    main_actions: List[Dict[str, Any]] = []
    winner_target = algo_targets.get(best_algo, {})

    if cfg.decision_mode.lower() == "best_algorithm":
        main_actions = rebalance_portfolio(
            portfolio=main,
            target_weights=winner_target if winner_target is not None else {},
            prices=prices,
            cfg=cfg,
            ts=ts,
            min_trade_notional=cfg.min_trade_notional,
        )

    # main equity history for UI
    mv = portfolio_value(main, prices)
    main.setdefault("equity_history", [])
    main["equity_history"].append({"ts": ts, "value": round(mv, 4), "cash": round(float(main.get("cash", 0.0)), 4)})

    # trim what UI shows to last simulation_hours (but keep bounded anyway)
    trim_equity_history_by_hours(main, cfg.simulation_hours)
    if len(main["equity_history"]) > cfg.max_history_points:
        main["equity_history"] = main["equity_history"][-cfg.max_history_points:]

    pnl = mv - float(state.get("starting_cash", cfg.starting_cash))
    pnl_pct = (pnl / float(state.get("starting_cash", cfg.starting_cash))) * 100.0 if float(state.get("starting_cash", cfg.starting_cash)) > 0 else 0.0

    # provide top coin picks for the best algo (for visibility)
    best_pack = packs.get(best_algo)
    best_scores = []
    if best_pack:
        for c, s in sorted(best_pack.scores.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            best_scores.append({"coin": c, "score": round(float(s), 4)})

    result = {
        "generated_at": ts,
        "prices_used": {k: round(float(v), 6) for k, v in prices.items()},

        "selected_algorithm": best_algo,
        "algorithm_leaderboard": leaderboard[:10],

        "winner_target_allocations": winner_target,
        "winner_top_coin_scores": best_scores,

        "main": {
            "value": round(float(mv), 4),
            "cash": round(float(main.get("cash", 0.0)), 4),
            "holdings": {k: round(float(v), 10) for k, v in (main.get("holdings") or {}).items()},
            "pnl": round(float(pnl), 4),
            "pnl_pct": round(float(pnl_pct), 4),
            "total_trades": int(main.get("total_trades", 0)),
            "tax_paid": round(float(main.get("tax_paid", 0.0)), 6),
        },

        "actions_this_run": main_actions[-50:],
        "equity_history": main.get("equity_history", [])[-min(len(main.get("equity_history", [])), cfg.max_history_points):],
    }

    return result
