"""Fetch MRT/LRT stations and bus stops in Singapore from OpenStreetMap (Overpass API).

No API key required. Results are cached as CSV in data/external/.
"""

import time
from pathlib import Path

import pandas as pd
import requests

EXTERNAL_DIR = Path(__file__).parents[2] / "data" / "external"

# Singapore bounding box: south, west, north, east
BBOX = "1.20,103.6,1.48,104.05"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

HEADERS = {"User-Agent": "hdb-price-analysis/1.0 (research)"}


def _run_query(query: str, retries: int = 3) -> list[dict]:
    """Run an Overpass query against mirrors with retry/backoff."""
    last_err = None
    for attempt in range(retries):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            resp = requests.post(
                endpoint, data={"data": query}, headers=HEADERS, timeout=180
            )
            if resp.status_code == 200:
                return resp.json()["elements"]
            last_err = f"HTTP {resp.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        wait = 5 * (attempt + 1)
        print(f"  Retry in {wait}s (endpoint failed: {last_err})…")
        time.sleep(wait)
    raise RuntimeError(f"Overpass query failed after {retries} attempts: {last_err}")


def fetch_mrt_stations(force: bool = False) -> pd.DataFrame:
    """Fetch MRT/LRT station nodes (name, lat, lon)."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL_DIR / "mrt_stations.csv"
    if path.exists() and not force:
        return pd.read_csv(path)

    query = f"""
    [out:json][timeout:120];
    (
      node["railway"="station"]({BBOX});
      node["railway"="halt"]({BBOX});
      node["station"="subway"]({BBOX});
    );
    out body;
    """
    print("Fetching MRT/LRT stations from Overpass…")
    elements = _run_query(query)
    rows = [
        {
            "name": e.get("tags", {}).get("name", "Unknown"),
            "lat": e["lat"],
            "lon": e["lon"],
        }
        for e in elements
        if "lat" in e and "lon" in e
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} stations → {path}")
    return df


def fetch_bus_stops(force: bool = False) -> pd.DataFrame:
    """Fetch bus stop nodes (name, lat, lon)."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL_DIR / "bus_stops.csv"
    if path.exists() and not force:
        return pd.read_csv(path)

    query = f"""
    [out:json][timeout:120];
    (
      node["highway"="bus_stop"]({BBOX});
    );
    out body;
    """
    print("Fetching bus stops from Overpass…")
    time.sleep(2)  # be gentle between queries
    elements = _run_query(query)
    rows = [
        {
            "name": e.get("tags", {}).get("name", "Unknown"),
            "lat": e["lat"],
            "lon": e["lon"],
        }
        for e in elements
        if "lat" in e and "lon" in e
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} bus stops → {path}")
    return df


if __name__ == "__main__":
    mrt = fetch_mrt_stations(force=True)
    bus = fetch_bus_stops(force=True)
    print(f"\nMRT/LRT stations: {len(mrt)}")
    print(f"Bus stops: {len(bus)}")
