import requests

BASE_URL = "https://apiv2.nobitex.ir"


def fetch_market_stats(dst_currency: str = "usdt") -> dict:
    # One request to respect rate limit (20/min). We request all markets quoted in dst_currency.
    url = f"{BASE_URL}/market/stats"
    r = requests.get(url, params={"dstCurrency": dst_currency}, timeout=25)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Nobitex API error: {data}")
    return data.get("stats", {})
