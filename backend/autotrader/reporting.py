import os
import json
import math
from datetime import datetime, timezone
from dataclasses import asdict


def _sanitize_for_json(x):
    # Make output strict-JSON (no NaN/Infinity) so browsers can parse it
    if x is None:
        return None

    # basic scalars
    if isinstance(x, bool) or isinstance(x, int) or isinstance(x, str):
        return x

    if isinstance(x, float):
        return x if math.isfinite(x) else None

    # numpy/pandas scalars (best-effort without importing numpy)
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

    # fallback: stringify unknown objects
    return str(x)


def write_results_bundle(result, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{ts}.json"

    payload = _sanitize_for_json(asdict(result))

    # write timestamped run
    with open(os.path.join(out_dir, run_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    # update latest
    with open(os.path.join(out_dir, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    # update index.json (list runs)
    runs = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("run_") and fn.endswith(".json"):
            runs.append(fn)

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "runs": runs[-200:],
                "latest": "latest.json",
            },
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
