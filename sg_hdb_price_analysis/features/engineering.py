"""Feature engineering for HDB resale price prediction."""

from pathlib import Path

import numpy as np
import pandas as pd

from sg_hdb_price_analysis.features.spatial import add_spatial_features


STOREY_RANGE_MAP = {
    "01 TO 03": 2, "04 TO 06": 5, "07 TO 09": 8, "10 TO 12": 11,
    "13 TO 15": 14, "16 TO 18": 17, "19 TO 21": 20, "22 TO 24": 23,
    "25 TO 27": 26, "28 TO 30": 29, "31 TO 33": 32, "34 TO 36": 35,
    "37 TO 39": 38, "40 TO 42": 41, "43 TO 45": 44, "46 TO 48": 47,
    "49 TO 51": 50,
}

FLAT_TYPE_ORDER = {
    "1 ROOM": 1, "2 ROOM": 2, "3 ROOM": 3, "4 ROOM": 4,
    "5 ROOM": 5, "EXECUTIVE": 6, "MULTI-GENERATION": 7,
}

# CBD reference point (Raffles Place) for distance-to-city-centre feature.
CBD_LAT, CBD_LON = 1.2830, 103.8513
_EARTH_R = 6_371_000.0

# Base year for the continuous monthly time index.
TIME_BASE_YEAR = 2017


def parse_remaining_lease(value) -> float:
    """Parse a OneMap-style 'NN years MM months' string to float years."""
    if pd.isna(value):
        return np.nan
    parts = str(value).split()
    years, months = 0.0, 0.0
    for i, tok in enumerate(parts):
        if tok.startswith("year"):
            years = float(parts[i - 1])
        elif tok.startswith("month"):
            months = float(parts[i - 1])
    return years + months / 12.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (vectorised)."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_R * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame, with_spatial: bool = True) -> pd.DataFrame:
    """Return feature matrix X and target y from raw dataframe.

    If with_spatial is True and geocoded coordinates exist, MRT/bus distance
    features are merged in.
    """
    df = df.copy()

    # Time features
    df["year"] = df["month"].dt.year
    df["month_of_year"] = df["month"].dt.month

    # Storey midpoint
    df["storey_mid"] = df["storey_range"].map(STOREY_RANGE_MAP)
    # Fallback: parse from string if not in map
    missing = df["storey_mid"].isna()
    if missing.any():
        parsed = df.loc[missing, "storey_range"].str.extract(r"(\d+)\s+TO\s+(\d+)")
        df.loc[missing, "storey_mid"] = (
            parsed[0].astype(float) + parsed[1].astype(float)
        ) / 2

    # Lease remaining years (rough estimate)
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"], errors="coerce")
    df["remaining_lease_years"] = df["lease_commence_date"].apply(
        lambda y: max(0, 99 - (df["year"].iloc[0] - y)) if pd.notna(y) else float("nan")
    )
    # Vectorised version
    df["remaining_lease_years"] = (
        df["lease_commence_date"] + 99 - df["year"]
    ).clip(lower=0)

    # Flat type ordinal
    df["flat_type_ord"] = df["flat_type"].map(FLAT_TYPE_ORDER)

    # Accurate remaining lease from the raw string; fall back to the estimate.
    if "remaining_lease" in df.columns:
        df["remaining_lease_exact"] = df["remaining_lease"].apply(parse_remaining_lease)
        df["remaining_lease_exact"] = df["remaining_lease_exact"].fillna(
            df["remaining_lease_years"]
        )
    else:
        df["remaining_lease_exact"] = df["remaining_lease_years"]

    # Flat age at time of sale.
    df["flat_age"] = (df["year"] - df["lease_commence_date"]).clip(lower=0)

    # Continuous monthly time index (captures the market trend smoothly).
    df["time_index"] = (df["year"] - TIME_BASE_YEAR) * 12 + df["month_of_year"]

    # Price per sqm (not used as feature — only for EDA)
    if "resale_price" in df.columns:
        df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]

    # Spatial features (nearest MRT/bus distance, counts within radius) + lat/lon
    coords_path = Path(__file__).parents[2] / "data" / "external" / "address_coords.csv"
    if with_spatial and coords_path.exists() and {"block", "street_name"}.issubset(df.columns):
        df = add_spatial_features(df, coords_path=coords_path)
        # Distance to CBD (needs merged lat/lon)
        if {"lat", "lon"}.issubset(df.columns):
            df["dist_cbd_m"] = haversine_m(df["lat"], df["lon"], CBD_LAT, CBD_LON)

    return df


CATEGORICAL_COLS = ["town", "flat_model"]

# Features always available from the record itself.
BASE_NUMERIC_FEATURES = [
    "floor_area_sqm",
    "storey_mid",
    "remaining_lease_exact",
    "flat_type_ord",
    "time_index",
    "flat_age",
]

# Coordinate-derived features (NaN for ungeocoded addresses → median-imputed).
COORD_FEATURES = [
    "lat",
    "lon",
    "dist_cbd_m",
    "dist_nearest_mrt_m",
    "dist_nearest_bus_m",
    "n_bus_400m",
]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + COORD_FEATURES

TARGET = "resale_price"
