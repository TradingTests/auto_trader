from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


ALGO_NAMES = [
    "Trend Following (Momentum)",
    "Mean Reversion",
    "Statistical Arbitrage",
    "Market Making",
    "High-Frequency Trading (HFT)",
    "Machine Learning / AI-Based Strategies",
    "Event-Driven Strategies",
    "Execution Algorithms (VWAP, TWAP, POV, IS)",
    "Multi-Factor / Factor Investing",
    "Volatility Arbitrage",
    "Cross-Exchange Arbitrage (Crypto)",
    "Liquidity Provision Bots (Crypto)",
    "Pairs Trading",
    "Basket Trading",
    "Reinforcement Learning Trading",
    "Sentiment/NLP-Based Trading",
    "Merger Arbitrage (Event-Driven subtype)",
]


@dataclass
class ScorePack:
    scores: Dict[str, float]     # coin -> score
    details: Dict[str, dict]     # coin -> debug details


def _clip(x: float, lo=-1.0, hi=1.0) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(max(lo, min(hi, x)))


def _zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


def _prep_panel(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if df.empty:
        return df, []

    df = df.copy()
    df = df.sort_values(["src", "ts"])

    for col in ["latest", "best_buy", "best_sell", "volume_src", "day_change"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume_src"] = df["volume_src"].fillna(0.0)
    df["day_change"] = df["day_change"].fillna(0.0)

    df["ret_1"] = df.groupby("src")["latest"].pct_change()
    df["ret_3"] = df.groupby("src")["latest"].pct_change(3)
    df["ret_12"] = df.groupby("src")["latest"].pct_change(12)   # ~1h if 5min snapshots
    df["ret_72"] = df.groupby("src")["latest"].pct_change(72)   # ~6h

    df["spread"] = (df["best_sell"] - df["best_buy"]) / df["latest"].replace(0, np.nan)

    df["vol_12"] = df.groupby("src")["ret_1"].rolling(12).std(ddof=0).reset_index(level=0, drop=True)
    df["vol_72"] = df.groupby("src")["ret_1"].rolling(72).std(ddof=0).reset_index(level=0, drop=True)

    df["price_z_36"] = df.groupby("src")["latest"].apply(lambda s: _zscore(s, 36)).reset_index(level=0, drop=True)

    coins = sorted(df["src"].astype(str).str.lower().unique().tolist())
    return df, coins


def trend_following_momentum(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores, details = {}, {}
    for c in coins:
        d = panel[panel["src"] == c].iloc[-1]
        mom = 0.6 * (d.get("ret_12") or 0.0) + 0.4 * (d.get("ret_72") or 0.0)
        scores[c] = _clip(5.0 * mom)
        details[c] = {"mom": float(mom)}
    return ScorePack(scores, details)


def mean_reversion(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores, details = {}, {}
    for c in coins:
        d = panel[panel["src"] == c].iloc[-1]
        z = d.get("price_z_36")
        z = 0.0 if z is None or not np.isfinite(z) else float(z)
        scores[c] = _clip(-0.35 * z)
        details[c] = {"z": z}
    return ScorePack(scores, details)


def statistical_arbitrage(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores, details = {}, {}
    last = panel.groupby("src").tail(1).set_index("src")
    rets = last["ret_12"].fillna(0.0)
    mu = float(rets.mean()) if len(rets) else 0.0
    for c in coins:
        dev = float(rets.get(c, 0.0) - mu)
        scores[c] = _clip(-6.0 * dev)
        details[c] = {"ret_12": float(rets.get(c, 0.0)), "cross_mean": mu}
    return ScorePack(scores, details)


def market_making(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores, details = {}, {}
    last = panel.groupby("src").tail(1).set_index("src")
    for c in coins:
        spr = float(last.loc[c, "spread"]) if c in last.index and pd.notna(last.loc[c, "spread"]) else 1.0
        vol = float(last.loc[c, "volume_src"]) if c in last.index else 0.0
        s = (0.5 * np.tanh(vol / 50.0)) + (0.5 * np.tanh((0.01 - spr) / 0.01))
        scores[c] = _clip(float(s))
        details[c] = {"spread": spr, "volume_src": vol}
    return ScorePack(scores, details)


def hft(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores, details = {}, {}
    for c in coins:
        sub = panel[panel["src"] == c].tail(4)
        r1 = float(sub["latest"].pct_change().iloc[-1]) if len(sub) >= 2 else 0.0
        r3 = float(sub["latest"].pct_change(3).iloc[-1]) if len(sub) >= 4 else 0.0
        scores[c] = _clip(10.0 * (0.7 * r1 + 0.3 * r3))
        details[c] = {"ret_1step": r1, "ret_3step": r3}
    return ScorePack(scores, details)


def ml_ai(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores, details = {}, {}
    k = 6
    for c in coins:
        sub = panel[panel["src"] == c].dropna(subset=["ret_1"]).tail(140)
        if len(sub) < (k + 30):
            scores[c] = 0.0
            details[c] = {"note": "not_enough_data"}
            continue

        rets = sub["ret_1"].values
        X, y = [], []
        for i in range(k, len(rets) - 1):
            X.append(rets[i - k:i])
            y.append(rets[i])
        X = np.asarray(X)
        y = np.asarray(y)
        if len(y) < 30:
            scores[c] = 0.0
            details[c] = {"note": "not_enough_samples"}
            continue

        model = LinearRegression()
        model.fit(X, y)
        pred = float(model.predict(rets[-k:].reshape(1, -1))[0])
        scores[c] = _clip(30.0 * pred)
        details[c] = {"pred_ret": pred}
    return ScorePack(scores, details)


def event_driven(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores, details = {}, {}
    last = panel.groupby("src").tail(1).set_index("src")
    for c in coins:
        dc = float(last.loc[c, "day_change"]) if c in last.index else 0.0
        scores[c] = _clip(0.03 * dc)
        details[c] = {"day_change_percent": dc}
    return ScorePack(scores, details)


def execution_algos(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    return ScorePack({c: 0.0 for c in coins}, {c: {"note": "modeled_via_slippage_and_tax"} for c in coins})


def multi_factor(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores, details = {}, {}
    last = panel.groupby("src").tail(1).set_index("src")
    vols = last["vol_72"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_rank = vols.rank(pct=True) if len(vols) else pd.Series()

    for c in coins:
        mom = float(last.loc[c, "ret_72"]) if c in last.index and pd.notna(last.loc[c, "ret_72"]) else 0.0
        v = float(last.loc[c, "volume_src"]) if c in last.index else 0.0
        vr = float(vol_rank.get(c, 0.5)) if len(vol_rank) else 0.5
        s = 4.0 * mom + 0.3 * np.tanh(v / 100.0) - 0.4 * (vr - 0.5)
        scores[c] = _clip(float(s))
        details[c] = {"mom_6h": mom, "vol_rank": vr, "volume_src": v}
    return ScorePack(scores, details)


def volatility_arbitrage(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores, details = {}, {}
    last = panel.groupby("src").tail(1).set_index("src")
    for c in coins:
        v12 = float(last.loc[c, "vol_12"]) if c in last.index and pd.notna(last.loc[c, "vol_12"]) else 0.0
        v72 = float(last.loc[c, "vol_72"]) if c in last.index and pd.notna(last.loc[c, "vol_72"]) else 0.0
        z = float(last.loc[c, "price_z_36"]) if c in last.index and pd.notna(last.loc[c, "price_z_36"]) else 0.0
        vol_spike = (v12 - v72)
        s = 2.0 * np.tanh(50.0 * vol_spike) + (-0.15 * z)
        scores[c] = _clip(float(s))
        details[c] = {"vol_12": v12, "vol_72": v72, "z": z}
    return ScorePack(scores, details)


def cross_exchange_arbitrage(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    return ScorePack({c: 0.0 for c in coins}, {c: {"note": "no_external_exchange_price_in_stats"} for c in coins})


def liquidity_provision(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    mm = market_making(panel, coins)
    scores = {c: float(0.7 * mm.scores[c]) for c in coins}
    details = {c: {"based_on": "spread_and_volume", **mm.details[c]} for c in coins}
    return ScorePack(scores, details)


def pairs_trading(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores = {c: 0.0 for c in coins}
    details = {c: {} for c in coins}

    N = 72
    piv = panel.pivot_table(index="ts", columns="src", values="latest").sort_index()
    if piv.shape[0] < N + 5 or piv.shape[1] < 2:
        for c in coins:
            details[c] = {"note": "not_enough_data"}
        return ScorePack(scores, details)

    rets = piv.pct_change().dropna().tail(N)
    corr = rets.corr().fillna(0.0)

    best = None
    best_val = -1.0
    cols = list(rets.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr.loc[cols[i], cols[j]])
            if val > best_val:
                best_val = val
                best = (cols[i], cols[j])

    if best is None:
        return ScorePack(scores, details)

    a, b = best
    spread = np.log(piv[a]) - np.log(piv[b])
    z = ((spread - spread.rolling(N).mean()) / spread.rolling(N).std(ddof=0)).iloc[-1]
    z = float(z) if np.isfinite(z) else 0.0

    scores[a] = _clip(-0.25 * z)
    scores[b] = _clip(+0.25 * z)
    details[a] = {"pair": b, "corr": best_val, "spread_z": z}
    details[b] = {"pair": a, "corr": best_val, "spread_z": z}
    return ScorePack(scores, details)


def basket_trading(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    return ScorePack({c: 0.0 for c in coins}, {c: {"note": "basket_is_allocation_method"} for c in coins})


def reinforcement_learning(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    scores, details = {}, {}
    for c in coins:
        sub = panel[panel["src"] == c].dropna(subset=["ret_1"]).tail(50)
        if len(sub) < 20:
            scores[c] = 0.0
            details[c] = {"note": "not_enough_data"}
            continue
        r = float(sub["ret_12"].iloc[-1]) if pd.notna(sub["ret_12"].iloc[-1]) else 0.0
        vol = float(sub["ret_1"].std(ddof=0))
        conf = 1.0 / (1.0 + 200.0 * vol)
        scores[c] = _clip(float(np.sign(r) * conf))
        details[c] = {"ret_1h": r, "conf": conf, "vol": vol}
    return ScorePack(scores, details)


def sentiment_nlp(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    return event_driven(panel, coins)


def merger_arbitrage(panel: pd.DataFrame, coins: List[str]) -> ScorePack:
    return ScorePack({c: 0.0 for c in coins}, {c: {"note": "not_applicable_in_crypto_stats"} for c in coins})


def compute_packs_by_algorithm(panel: pd.DataFrame) -> Dict[str, ScorePack]:
    panel, coins = _prep_panel(panel)
    if panel.empty or not coins:
        return {}

    packs: Dict[str, ScorePack] = {}
    packs["Trend Following (Momentum)"] = trend_following_momentum(panel, coins)
    packs["Mean Reversion"] = mean_reversion(panel, coins)
    packs["Statistical Arbitrage"] = statistical_arbitrage(panel, coins)
    packs["Market Making"] = market_making(panel, coins)
    packs["High-Frequency Trading (HFT)"] = hft(panel, coins)
    packs["Machine Learning / AI-Based Strategies"] = ml_ai(panel, coins)
    packs["Event-Driven Strategies"] = event_driven(panel, coins)
    packs["Execution Algorithms (VWAP, TWAP, POV, IS)"] = execution_algos(panel, coins)
    packs["Multi-Factor / Factor Investing"] = multi_factor(panel, coins)
    packs["Volatility Arbitrage"] = volatility_arbitrage(panel, coins)
    packs["Cross-Exchange Arbitrage (Crypto)"] = cross_exchange_arbitrage(panel, coins)
    packs["Liquidity Provision Bots (Crypto)"] = liquidity_provision(panel, coins)
    packs["Pairs Trading"] = pairs_trading(panel, coins)
    packs["Basket Trading"] = basket_trading(panel, coins)
    packs["Reinforcement Learning Trading"] = reinforcement_learning(panel, coins)
    packs["Sentiment/NLP-Based Trading"] = sentiment_nlp(panel, coins)
    packs["Merger Arbitrage (Event-Driven subtype)"] = merger_arbitrage(panel, coins)
    return packs
