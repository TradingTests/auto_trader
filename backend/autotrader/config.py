from dataclasses import dataclass
from typing import List, Dict
import yaml


@dataclass
class Config:
    quote_currency: str
    coins: List[str]
    starting_cash: float
    tax_percent: float
    slippage_bps: float

    simulation_hours: int
    rebalance_every_minutes: int

    max_positions: int
    min_score_to_invest: float

    decision_mode: str
    algo_min_score_to_invest: float
    algo_rank_lookback_points: int

    min_trade_notional: float

    snapshot_retention_days: int
    max_history_points: int

    strategy_weights: Dict[str, float]

    @property
    def tax_rate(self) -> float:
        return float(self.tax_percent) / 100.0

    @property
    def slippage_rate(self) -> float:
        return float(self.slippage_bps) / 10_000.0


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}

    return Config(
        quote_currency=str(d.get("quote_currency", "usdt")).lower(),
        coins=[str(x).lower() for x in d.get("coins", [])],

        starting_cash=float(d.get("starting_cash", 1000)),
        tax_percent=float(d.get("tax_percent", 0.2)),
        slippage_bps=float(d.get("slippage_bps", 5)),

        simulation_hours=int(d.get("simulation_hours", 24)),
        rebalance_every_minutes=int(d.get("rebalance_every_minutes", 5)),

        max_positions=int(d.get("max_positions", 3)),
        min_score_to_invest=float(d.get("min_score_to_invest", 0.02)),

        decision_mode=str(d.get("decision_mode", "best_algorithm")),
        algo_min_score_to_invest=float(d.get("algo_min_score_to_invest", -1.0)),
        algo_rank_lookback_points=int(d.get("algo_rank_lookback_points", 288)),

        min_trade_notional=float(d.get("min_trade_notional", 5)),

        snapshot_retention_days=int(d.get("snapshot_retention_days", 14)),
        max_history_points=int(d.get("max_history_points", 4000)),

        strategy_weights=dict(d.get("strategy_weights", {})),
    )
