import os
import json
import math
from datetime import datetime, timezone


def _sanitize(x):
    if x is None:
        return None
    if isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    t = type(x).__name__.lower()
    if "float" in t or "int" in t:
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None
    if isinstance(x, dict):
        return {str(k): _sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize(v) for v in x]
    return str(x)


def write_results(result: dict, out_dir: str = "backend/results", history_cap: int = 2000):
    os.makedirs(out_dir, exist_ok=True)

    result = _sanitize(result)

    latest_path = os.path.join(out_dir, "latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)

    # compact history for long-term plotting
    hist_path = os.path.join(out_dir, "history.json")
    hist = []
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            if not isinstance(hist, list):
                hist = []
        except Exception:
            hist = []

    hist.append({
        "ts": result.get("generated_at"),
        "value": (result.get("main") or {}).get("value"),
        "pnl_pct": (result.get("main") or {}).get("pnl_pct"),
        "selected_algorithm": result.get("selected_algorithm"),
        "actions": len(result.get("actions_this_run") or []),
    })

    if len(hist) > history_cap:
        hist = hist[-history_cap:]

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=1, allow_nan=False)

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "latest": "latest.json",
                "history": "history.json",
                "history_points": len(hist),
            },
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
