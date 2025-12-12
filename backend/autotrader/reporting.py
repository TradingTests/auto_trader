import os
import json
import math
from datetime import datetime, timezone
from dataclasses import asdict


def _sanitize_for_json(x):
    """Make output strict-JSON (no NaN/Infinity)"""
    if x is None:
        return None

    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        return x

    if isinstance(x, float):
        return x if math.isfinite(x) else None

    # numpy/pandas scalars
    t = type(x).__name__.lower()
    if "float" in t or "int" in t:
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None

    if isinstance(x, dict):
        return {str(k): _sanitize_for_json(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_sanitize_for_json(v) for v in x]

    return str(x)


def write_results(result, out_dir: str = "backend/results"):
    """
    Write only TWO files:
    - latest.json: Current state (overwritten each run)
    - history.json: Append-only daily summaries (compact)
    """
    os.makedirs(out_dir, exist_ok=True)

    payload = _sanitize_for_json(asdict(result))

    # 1) Always overwrite latest.json
    latest_path = os.path.join(out_dir, "latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    # 2) Append to daily history (one entry per run, but capped)
    history_path = os.path.join(out_dir, "history.json")
    
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
    
    # Compact summary entry
    summary_entry = {
        "ts": result.generated_at if hasattr(result, 'generated_at') else datetime.now(timezone.utc).isoformat(),
        "value": payload.get("portfolio_value"),
        "cash": payload.get("cash"),
        "pnl_pct": payload.get("pnl_pct"),
        "trades": payload.get("total_trades"),
        "actions": len(payload.get("actions_this_run", [])),
        "holdings": list(payload.get("holdings", {}).keys()),
    }
    
    history.append(summary_entry)
    
    # Keep only last 2000 entries (~1 week at 5min intervals)
    if len(history) > 2000:
        history = history[-2000:]
    
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1, allow_nan=False)

    # 3) Write simple index
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "latest": "latest.json",
            "history": "history.json",
            "total_history_points": len(history),
        }, f, ensure_ascii=False, indent=2, allow_nan=False)
