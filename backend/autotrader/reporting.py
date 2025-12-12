import os
import json
from datetime import datetime, timezone
from dataclasses import asdict
from .simulator import SimulationResult


def write_results_bundle(result: SimulationResult, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{ts}.json"

    payload = asdict(result)

    # write timestamped run
    with open(os.path.join(out_dir, run_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # update latest
    with open(os.path.join(out_dir, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # update index.json (list runs)
    runs = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("run_") and fn.endswith(".json"):
            runs.append(fn)

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "runs": runs[-200:],  # keep last 200
                "latest": "latest.json",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
