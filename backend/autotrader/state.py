import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List
from datetime import datetime, timezone


STATE_FILE = "backend/data/portfolio_state.json"


@dataclass
class PortfolioState:
    """Persisted across runs - this is the 'memory' of the trader"""
    
    # Current holdings
    cash: float = 1000.0
    holdings: Dict[str, float] = field(default_factory=dict)  # coin -> quantity
    
    # Cumulative stats
    total_trades: int = 0
    total_tax_paid: float = 0.0
    total_realized_pnl: float = 0.0
    
    # Track performance
    initial_capital: float = 1000.0
    high_water_mark: float = 1000.0
    max_drawdown_pct: float = 0.0
    
    # History (compact - just daily snapshots for charts)
    equity_history: List[dict] = field(default_factory=list)  # [{ts, value}, ...]
    
    # Last update
    last_run_at: str = ""
    last_prices: Dict[str, float] = field(default_factory=dict)
    
    # Algorithm performance tracking (which strategies are winning)
    strategy_pnl: Dict[str, float] = field(default_factory=dict)


def load_state(starting_cash: float) -> PortfolioState:
    """Load existing state or create new one"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            state = PortfolioState(
                cash=float(data.get("cash", starting_cash)),
                holdings={str(k): float(v) for k, v in data.get("holdings", {}).items()},
                total_trades=int(data.get("total_trades", 0)),
                total_tax_paid=float(data.get("total_tax_paid", 0.0)),
                total_realized_pnl=float(data.get("total_realized_pnl", 0.0)),
                initial_capital=float(data.get("initial_capital", starting_cash)),
                high_water_mark=float(data.get("high_water_mark", starting_cash)),
                max_drawdown_pct=float(data.get("max_drawdown_pct", 0.0)),
                equity_history=list(data.get("equity_history", [])),
                last_run_at=str(data.get("last_run_at", "")),
                last_prices={str(k): float(v) for k, v in data.get("last_prices", {}).items()},
                strategy_pnl={str(k): float(v) for k, v in data.get("strategy_pnl", {}).items()},
            )
            return state
        except Exception as e:
            print(f"Warning: Could not load state file, starting fresh: {e}")
    
    # Fresh start
    return PortfolioState(
        cash=starting_cash,
        initial_capital=starting_cash,
        high_water_mark=starting_cash,
    )


def save_state(state: PortfolioState):
    """Save state to disk"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    
    # Keep equity history compact (max 2000 points ~= 1 week at 5min intervals)
    if len(state.equity_history) > 2000:
        # Downsample: keep every 12th point (hourly) for old data
        old = state.equity_history[:-500]
        recent = state.equity_history[-500:]
        downsampled = old[::12]
        state.equity_history = downsampled + recent
    
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, ensure_ascii=False, indent=2)


def get_portfolio_value(state: PortfolioState, prices: Dict[str, float]) -> float:
    """Calculate total portfolio value at current prices"""
    value = state.cash
    for coin, qty in state.holdings.items():
        px = prices.get(coin, state.last_prices.get(coin, 0))
        if px and px > 0:
            value += qty * px
    return value
