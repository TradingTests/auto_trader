#!/usr/bin/env python3
"""
AutoTrader Backend Runner
- Fetches market data
- Updates persistent portfolio state
- Executes trading decisions
- Saves results
"""

from autotrader.config import load_config
from autotrader.nobitex import fetch_market_stats
from autotrader.storage import append_snapshots, load_recent_history
from autotrader.state import load_state, save_state
from autotrader.simulator import run_simulation
from autotrader.reporting import write_results


def main():
    print("=" * 50)
    print("AutoTrader Backend Run")
    print("=" * 50)
    
    # 1) Load config
    cfg = load_config("config.yaml")
    print(f"Config loaded: {len(cfg.coins)} coins, starting_cash={cfg.starting_cash}")
    
    # 2) Load persistent portfolio state
    state = load_state(cfg.starting_cash)
    print(f"Portfolio state: cash={state.cash:.2f}, holdings={list(state.holdings.keys())}, trades={state.total_trades}")
    
    # 3) Fetch current market data
    try:
        stats = fetch_market_stats(dst_currency=cfg.quote_currency)
        print(f"Fetched {len(stats)} market pairs from Nobitex")
    except Exception as e:
        print(f"ERROR fetching market data: {e}")
        return
    
    # 4) Store snapshot
    append_snapshots(
        stats=stats,
        quote_currency=cfg.quote_currency,
        allowed_src=set(cfg.coins),
        csv_path="backend/data/market_snapshots.csv",
    )
    
    # 5) Load recent history for analysis
    history = load_recent_history(
        csv_path="backend/data/market_snapshots.csv",
        hours=cfg.simulation_hours,
        allowed_src=set(cfg.coins),
        quote_currency=cfg.quote_currency,
    )
    print(f"Loaded {len(history)} history rows (last {cfg.simulation_hours}h)")
    
    # 6) Run simulation / make trading decisions
    result = run_simulation(cfg, history, state)
    
    # 7) Save updated state
    save_state(state)
    print(f"State saved: value={result.portfolio_value:.2f}, pnl={result.pnl_pct:.2f}%")
    
    # 8) Write results for frontend
    write_results(result)
    
    # 9) Print summary
    print("-" * 50)
    print(f"Portfolio Value: ${result.portfolio_value:.2f}")
    print(f"P&L: ${result.pnl_total:.2f} ({result.pnl_pct:.2f}%)")
    print(f"Total Trades: {result.total_trades}")
    print(f"Actions this run: {len(result.actions_this_run)}")
    if result.actions_this_run:
        for a in result.actions_this_run:
            print(f"  {a['action']} {a['coin']}: {a['qty']:.6f} @ ${a['price']:.2f}")
    print(f"Holdings: {result.holdings}")
    print("=" * 50)


if __name__ == "__main__":
    main()
