from autotrader.config import load_config
from autotrader.nobitex import fetch_market_stats
from autotrader.storage import append_snapshots, load_recent_history
from autotrader.state import load_state, save_state
from autotrader.simulator import run_step
from autotrader.reporting import write_results


def main():
    cfg = load_config("config.yaml")
    state = load_state(cfg.starting_cash)

    stats = fetch_market_stats(dst_currency=cfg.quote_currency)

    append_snapshots(
        stats=stats,
        quote_currency=cfg.quote_currency,
        allowed_src=set(cfg.coins),
        csv_path="backend/data/market_snapshots.csv",
        retention_days=cfg.snapshot_retention_days,
    )

    history = load_recent_history(
        csv_path="backend/data/market_snapshots.csv",
        hours=cfg.simulation_hours,
        allowed_src=set(cfg.coins),
        quote_currency=cfg.quote_currency,
    )

    result = run_step(cfg, history, state)

    # persist state + results
    save_state(state, max_history_points=cfg.max_history_points)
    write_results(result)


if __name__ == "__main__":
    main()
