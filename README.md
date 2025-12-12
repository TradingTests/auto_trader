# Auto_Trader

Simulation-only crypto auto-trader:
- Backend fetches Nobitex market stats, stores snapshots, runs multi-strategy simulation, writes JSON result files.
- Frontend is a static page that displays the result JSON files.
https://tradingtests.github.io/auto_trader/frontend/

## Run locally

```bash
pip install -r requirements.txt
python backend/run_backend.py
python -m http.server 8000
