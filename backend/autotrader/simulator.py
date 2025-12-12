from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

from .config import Config
from .strategies import compute_all_scores
from .state import PortfolioState, get_portfolio_value


@dataclass
class TradeAction:
    ts: str
    action: str  # BUY, SELL, HOLD
    coin: str
    qty: float
    price: float
    notional: float
    tax: float
    reason: str
    score: float = 0.0


@dataclass 
class SimulationResult:
    generated_at: str
    portfolio_value: float
    cash: float
    holdings: Dict[str, float]
    pnl_total: float
    pnl_pct: float
    high_water_mark: float
    max_drawdown_pct: float
    total_trades: int
    total_tax_paid: float
    
    # What happened this run
    actions_this_run: List[dict]
    scores: List[dict]
    target_allocations: Dict[str, float]
    
    # For charts
    equity_history: List[dict]
    
    # Strategy performance
    strategy_contributions: Dict[str, float]
    
    # Debug
    debug: Dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _get_current_prices(df: pd.DataFrame) -> Dict[str, float]:
    """Get latest price for each coin from the most recent snapshot"""
    if df.empty:
        return {}
    
    latest = df.sort_values("ts").groupby("src").tail(1)
    return {str(row["src"]): float(row["latest"]) for _, row in latest.iterrows()}


def _select_targets(
    scores: Dict[str, float], 
    max_positions: int, 
    min_score: float,
    prices: Dict[str, float]
) -> Dict[str, float]:
    """
    Select target allocation weights based on scores.
    Returns {coin: weight} where weights sum to <= 1.0 (rest is cash)
    """
    # Filter coins with valid prices and good scores
    valid = [(c, s) for c, s in scores.items() if s >= min_score and prices.get(c, 0) > 0]
    valid = sorted(valid, key=lambda x: x[1], reverse=True)[:max_positions]
    
    if not valid:
        return {}  # Stay in cash
    
    # Weight by score (higher score = more allocation)
    total_score = sum(s for _, s in valid)
    if total_score <= 0:
        # Equal weight if all scores are similar
        w = 1.0 / len(valid)
        return {c: w for c, _ in valid}
    
    # Score-weighted allocation
    weights = {}
    for c, s in valid:
        # Normalize and ensure positive weights
        w = max(0.05, s / total_score)  # min 5% if selected
        weights[c] = w
    
    # Normalize to sum to 1.0
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()}


def execute_rebalance(
    state: PortfolioState,
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    scores: Dict[str, float],
    cfg: Config,
) -> Tuple[List[TradeAction], Dict[str, float]]:
    """
    Execute trades to move from current holdings to target weights.
    Returns list of trades and strategy contribution estimates.
    """
    actions = []
    strategy_contrib = {}
    
    current_value = get_portfolio_value(state, prices)
    if current_value <= 0:
        return actions, strategy_contrib
    
    # Calculate current weights
    current_weights = {"cash": state.cash / current_value}
    for coin, qty in state.holdings.items():
        px = prices.get(coin, 0)
        if px > 0:
            current_weights[coin] = (qty * px) / current_value
    
    ts = _now_iso()
    
    # 1) SELL coins not in target or overweight
    for coin in list(state.holdings.keys()):
        current_w = current_weights.get(coin, 0)
        target_w = target_weights.get(coin, 0)
        
        if target_w < current_w:
            # Need to sell some or all
            px = prices.get(coin, 0)
            if px <= 0:
                continue
            
            current_qty = state.holdings[coin]
            target_qty = (target_w * current_value) / px
            sell_qty = current_qty - target_qty
            
            if sell_qty < 0.0000001:
                continue
            
            # Execute sell with slippage
            exec_px = px * (1.0 - cfg.slippage_rate)
            proceeds = sell_qty * exec_px
            tax = proceeds * cfg.tax_rate
            net_proceeds = proceeds - tax
            
            # Calculate realized PnL (simplified - avg cost basis would be better)
            
            # Update state
            state.cash += net_proceeds
            state.holdings[coin] -= sell_qty
            if state.holdings[coin] < 0.0000001:
                del state.holdings[coin]
            
            state.total_trades += 1
            state.total_tax_paid += tax
            
            reason = "exit_position" if target_w == 0 else "reduce_position"
            actions.append(TradeAction(
                ts=ts,
                action="SELL",
                coin=coin,
                qty=sell_qty,
                price=exec_px,
                notional=proceeds,
                tax=tax,
                reason=reason,
                score=scores.get(coin, 0),
            ))
    
    # Recalculate value after sells
    current_value = get_portfolio_value(state, prices)
    
    # 2) BUY coins in target or underweight
    for coin, target_w in target_weights.items():
        px = prices.get(coin, 0)
        if px <= 0:
            continue
        
        current_qty = state.holdings.get(coin, 0)
        current_coin_value = current_qty * px
        target_value = target_w * current_value
        buy_value = target_value - current_coin_value
        
        if buy_value < 1.0:  # Min $1 trade
            continue
        
        # Check available cash
        exec_px = px * (1.0 + cfg.slippage_rate)
        buy_qty = buy_value / exec_px
        cost = buy_qty * exec_px
        tax = cost * cfg.tax_rate
        total_cost = cost + tax
        
        if total_cost > state.cash:
            # Scale down to available cash
            scale = (state.cash * 0.99) / total_cost  # Keep 1% buffer
            if scale < 0.1:
                continue
            buy_qty *= scale
            cost = buy_qty * exec_px
            tax = cost * cfg.tax_rate
            total_cost = cost + tax
        
        if buy_qty < 0.0000001:
            continue
        
        # Execute buy
        state.cash -= total_cost
        state.holdings[coin] = state.holdings.get(coin, 0) + buy_qty
        state.total_trades += 1
        state.total_tax_paid += tax
        
        reason = "new_position" if current_qty == 0 else "increase_position"
        actions.append(TradeAction(
            ts=ts,
            action="BUY",
            coin=coin,
            qty=buy_qty,
            price=exec_px,
            notional=cost,
            tax=tax,
            reason=reason,
            score=scores.get(coin, 0),
        ))
        
        # Track which strategies contributed to this buy
        strategy_contrib[coin] = scores.get(coin, 0)
    
    return actions, strategy_contrib


def run_simulation(cfg: Config, history: pd.DataFrame, state: PortfolioState) -> SimulationResult:
    """
    Run one iteration of the trading simulation.
    Uses and updates the persistent portfolio state.
    """
    ts_now = _now_iso()
    
    if history.empty:
        return SimulationResult(
            generated_at=ts_now,
            portfolio_value=state.cash,
            cash=state.cash,
            holdings=dict(state.holdings),
            pnl_total=0,
            pnl_pct=0,
            high_water_mark=state.high_water_mark,
            max_drawdown_pct=0,
            total_trades=state.total_trades,
            total_tax_paid=state.total_tax_paid,
            actions_this_run=[],
            scores=[],
            target_allocations={},
            equity_history=state.equity_history,
            strategy_contributions={},
            debug={"note": "no_market_data"},
        )
    
    # Get current prices
    prices = _get_current_prices(history)
    state.last_prices = prices
    
    # Calculate scores using all available history
    scores, score_details = compute_all_scores(history, cfg.strategy_weights)
    
    if not scores:
        # No valid scores, hold current positions
        current_value = get_portfolio_value(state, prices)
        return SimulationResult(
            generated_at=ts_now,
            portfolio_value=current_value,
            cash=state.cash,
            holdings=dict(state.holdings),
            pnl_total=current_value - state.initial_capital,
            pnl_pct=((current_value / state.initial_capital) - 1) * 100,
            high_water_mark=state.high_water_mark,
            max_drawdown_pct=state.max_drawdown_pct,
            total_trades=state.total_trades,
            total_tax_paid=state.total_tax_paid,
            actions_this_run=[],
            scores=[],
            target_allocations={},
            equity_history=state.equity_history,
            strategy_contributions={},
            debug={"note": "no_valid_scores"},
        )
    
    # Determine target allocation
    target_weights = _select_targets(
        scores=scores,
        max_positions=cfg.max_positions,
        min_score=cfg.min_score_to_invest,
        prices=prices,
    )
    
    # Execute rebalance
    actions, strategy_contrib = execute_rebalance(
        state=state,
        target_weights=target_weights,
        prices=prices,
        scores=scores,
        cfg=cfg,
    )
    
    # Update strategy PnL tracking
    for coin, contrib in strategy_contrib.items():
        state.strategy_pnl[coin] = state.strategy_pnl.get(coin, 0) + contrib
    
    # Calculate final portfolio value
    current_value = get_portfolio_value(state, prices)
    
    # Update high water mark and drawdown
    if current_value > state.high_water_mark:
        state.high_water_mark = current_value
    
    drawdown = (state.high_water_mark - current_value) / state.high_water_mark * 100
    if drawdown > state.max_drawdown_pct:
        state.max_drawdown_pct = drawdown
    
    # Add to equity history
    state.equity_history.append({
        "ts": ts_now,
        "value": round(current_value, 2),
        "cash": round(state.cash, 2),
    })
    
    state.last_run_at = ts_now
    
    # Prepare scores for output
    scores_list = sorted(
        [{"coin": c, "score": round(s, 4), "details": score_details.get(c, {})} 
         for c, s in scores.items()],
        key=lambda x: x["score"],
        reverse=True,
    )
    
    pnl_total = current_value - state.initial_capital
    pnl_pct = ((current_value / state.initial_capital) - 1) * 100 if state.initial_capital > 0 else 0
    
    return SimulationResult(
        generated_at=ts_now,
        portfolio_value=round(current_value, 2),
        cash=round(state.cash, 2),
        holdings={c: round(q, 8) for c, q in state.holdings.items()},
        pnl_total=round(pnl_total, 2),
        pnl_pct=round(pnl_pct, 2),
        high_water_mark=round(state.high_water_mark, 2),
        max_drawdown_pct=round(state.max_drawdown_pct, 2),
        total_trades=state.total_trades,
        total_tax_paid=round(state.total_tax_paid, 2),
        actions_this_run=[{
            "ts": a.ts,
            "action": a.action,
            "coin": a.coin,
            "qty": round(a.qty, 8),
            "price": round(a.price, 2),
            "notional": round(a.notional, 2),
            "tax": round(a.tax, 4),
            "reason": a.reason,
            "score": round(a.score, 4),
        } for a in actions],
        scores=scores_list[:15],  # Top 15 only
        target_allocations={c: round(w, 4) for c, w in target_weights.items()},
        equity_history=state.equity_history[-500:],  # Last 500 for JSON
        strategy_contributions=strategy_contrib,
        debug={
            "prices": {c: round(p, 2) for c, p in list(prices.items())[:10]},
        },
    )
