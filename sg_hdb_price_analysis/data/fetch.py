"""Fetch HDB resale price data from data.gov.sg API."""

import time
from pathlib import Path

import pandas as pd
import requests

# Dataset IDs on data.gov.sg (verified working as of 2024)
# Older datasets (pre-2017) use different resource IDs that may change over time.
DATASET_IDS = [
    ("d_8b84c4ee58e3cfc0ece0d773c8ca6abc", "2017-present"),
    # Add older IDs here if found; the script will skip 404s automatically.
]

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"


def fetch_dataset(dataset_id: str, label: str, limit: int = 10000) -> pd.DataFrame | None:
    """Fetch all records for a single dataset via pagination. Returns None on 404."""
    url = "https://data.gov.sg/api/action/datastore_search"
    records = []
    offset = 0

    while True:
        resp = requests.get(
            url,
            params={"resource_id": dataset_id, "limit": limit, "offset": offset},
            timeout=30,
        )
        if resp.status_code == 404:
            print(f"  [{label}] Dataset not found (404), skipping.")
            return None
        resp.raise_for_status()
        result = resp.json()["result"]
        batch = result["records"]
        records.extend(batch)
        print(f"  [{label}] offset={offset:>7,} | rows so far: {len(records):>7,}")

        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.3)

    return pd.DataFrame(records)


def fetch_all(force: bool = False) -> pd.DataFrame:
    """Fetch and concatenate all datasets, caching to CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / "hdb_resale_prices_all.csv"

    if cache_path.exists() and not force:
        print(f"Loading from cache: {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["month"])

    dfs = []
    for dataset_id, label in DATASET_IDS:
        print(f"Fetching: {label} ({dataset_id[:12]}…)")
        df = fetch_dataset(dataset_id, label)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No data fetched. Check your dataset IDs.")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop(columns=["_id"], errors="ignore")

    combined["resale_price"] = pd.to_numeric(combined["resale_price"], errors="coerce")
    combined["floor_area_sqm"] = pd.to_numeric(combined["floor_area_sqm"], errors="coerce")
    combined["month"] = pd.to_datetime(combined["month"])

    combined = combined.sort_values("month").reset_index(drop=True)
    combined.to_csv(cache_path, index=False)
    print(f"\nSaved {len(combined):,} rows → {cache_path}")
    return combined


if __name__ == "__main__":
    df = fetch_all(force=True)
    print(df.dtypes)
    print(df.tail())
