"""Geocode unique HDB addresses to lat/lon using the free OneMap search API.

Processes only unique (block, street_name) pairs, runs in a thread pool, and
caches incrementally so the job is resumable.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import threading
import time

import pandas as pd
import requests

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
EXTERNAL_DIR = Path(__file__).parents[2] / "data" / "external"
CACHE_PATH = EXTERNAL_DIR / "address_coords.csv"

ONEMAP_URL = "https://www.onemap.gov.sg/api/common/elastic/search"
# Optional OneMap API token (JWT). A valid token raises the rate limit.
ONEMAP_TOKEN = os.environ.get("ONEMAP_TOKEN", "").strip()
_HEADERS = {"Authorization": f"Bearer {ONEMAP_TOKEN}"} if ONEMAP_TOKEN else {}
_lock = threading.Lock()


def geocode_one(address: str, retries: int = 4) -> tuple[float | None, float | None]:
    """Return (lat, lon) for an address, or (None, None) if not found.

    Retries with backoff on empty results (OneMap throttles under load).
    """
    for attempt in range(retries):
        try:
            resp = requests.get(
                ONEMAP_URL,
                params={
                    "searchVal": address,
                    "returnGeom": "Y",
                    "getAddrDetails": "Y",
                    "pageNum": 1,
                },
                headers=_HEADERS,
                timeout=20,
            )
            results = resp.json().get("results", [])
            if results and results[0].get("LATITUDE"):
                r = results[0]
                return float(r["LATITUDE"]), float(r["LONGITUDE"])
            # Empty → likely throttled; back off and retry
            time.sleep(0.5 * (attempt + 1))
        except Exception:  # noqa: BLE001
            time.sleep(0.5 * (attempt + 1))
    return None, None


def get_unique_addresses() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "hdb_resale_prices_all.csv", usecols=["block", "street_name"])
    df["address"] = df["block"].astype(str).str.strip() + " " + df["street_name"].astype(str).str.strip()
    uniq = df[["block", "street_name", "address"]].drop_duplicates("address").reset_index(drop=True)
    return uniq


def geocode_all(max_workers: int = 2, force: bool = False) -> pd.DataFrame:
    """Geocode all unique HDB addresses, caching incrementally."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    uniq = get_unique_addresses()

    done: dict[str, tuple[float, float]] = {}
    if CACHE_PATH.exists() and not force:
        cached = pd.read_csv(CACHE_PATH)
        done = {
            row.address: (row.lat, row.lon)
            for row in cached.itertuples()
            if pd.notna(row.lat)
        }
        print(f"Resuming: {len(done):,} addresses already cached.")

    todo = [a for a in uniq["address"] if a not in done]
    print(f"Geocoding {len(todo):,} addresses ({max_workers} workers)…")

    results = dict(done)
    counter = {"n": 0}

    def worker(addr: str):
        lat, lon = geocode_one(addr)
        return addr, lat, lon

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, a): a for a in todo}
        for fut in as_completed(futures):
            addr, lat, lon = fut.result()
            results[addr] = (lat, lon)
            with _lock:
                counter["n"] += 1
                if counter["n"] % 200 == 0:
                    print(f"  {counter['n']:,}/{len(todo):,} done", flush=True)
                    _save(results)

    _save(results)
    out = pd.read_csv(CACHE_PATH)
    found = out["lat"].notna().sum()
    print(f"\nGeocoded {found:,}/{len(out):,} addresses ({found / len(out) * 100:.1f}% success).")
    return out


def _save(results: dict):
    rows = [{"address": a, "lat": v[0], "lon": v[1]} for a, v in results.items()]
    pd.DataFrame(rows).to_csv(CACHE_PATH, index=False)


if __name__ == "__main__":
    geocode_all(force=False)
