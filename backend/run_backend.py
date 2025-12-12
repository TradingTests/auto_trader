from autotrader.config import load_config
from autotrader.nobitex import fetch_market_stats
from autotrader.storage import append_snapshots, load_recent_history
from autotrader.simulator import run_simulation
from autotrader.reporting import write_results_bundle


def main():
    cfg = load_config("config.yaml")

    stats = fetch_market_stats(dst_currency=cfg.quote_currency)
    append_snapshots(
        stats=stats,
        quote_currency=cfg.quote_currency,
        allowed_src=set(cfg.coins),
        csv_path="backend/data/market_snapshots.csv",
    )

    history = load_recent_history(
        csv_path="backend/data/market_snapshots.csv",
        hours=cfg.simulation_hours,
        allowed_src=set(cfg.coins),
        quote_currency=cfg.quote_currency,
    )

    result = run_simulation(cfg, history)
    write_results_bundle(result, out_dir="backend/results")


if __name__ == "__main__":
    main()
