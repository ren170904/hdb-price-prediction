"""Spatial features: distance to nearest MRT/bus stop and counts within a radius."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EXTERNAL_DIR = Path(__file__).parents[2] / "data" / "external"
EARTH_RADIUS_M = 6_371_000.0


def _load_points(filename: str) -> np.ndarray:
    """Load lat/lon points (radians) for BallTree haversine queries."""
    df = pd.read_csv(EXTERNAL_DIR / filename)
    coords = df[["lat", "lon"]].dropna().to_numpy()
    return np.radians(coords)


def _nearest_distance_and_count(
    query_coords: np.ndarray, ref_coords_rad: np.ndarray, radius_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (nearest distance in metres, count within radius) for each query point."""
    tree = BallTree(ref_coords_rad, metric="haversine")
    q_rad = np.radians(query_coords)

    dist_rad, _ = tree.query(q_rad, k=1)
    nearest_m = dist_rad[:, 0] * EARTH_RADIUS_M

    radius_rad = radius_m / EARTH_RADIUS_M
    counts = tree.query_radius(q_rad, r=radius_rad, count_only=True)

    return nearest_m, counts


def add_spatial_features(
    df: pd.DataFrame,
    coords_path: Path | None = None,
    mrt_radius_m: float = 1000.0,
    bus_radius_m: float = 400.0,
) -> pd.DataFrame:
    """Merge geocoded coordinates and compute MRT/bus distance features.

    Requires df to have 'block' and 'street_name' columns. Rows whose address
    could not be geocoded get NaN coordinates and distance features.
    """
    df = df.copy()
    coords_path = coords_path or (EXTERNAL_DIR / "address_coords.csv")

    coords = pd.read_csv(coords_path)
    df["address"] = (
        df["block"].astype(str).str.strip() + " " + df["street_name"].astype(str).str.strip()
    )
    df = df.merge(coords[["address", "lat", "lon"]], on="address", how="left")

    has_coord = df["lat"].notna() & df["lon"].notna()
    query = df.loc[has_coord, ["lat", "lon"]].to_numpy()

    # Initialise feature columns
    for col in ["dist_nearest_mrt_m", "dist_nearest_bus_m", "n_mrt_1km", "n_bus_400m"]:
        df[col] = np.nan

    if len(query) > 0:
        mrt_rad = _load_points("mrt_stations.csv")
        bus_rad = _load_points("bus_stops.csv")

        mrt_dist, mrt_cnt = _nearest_distance_and_count(query, mrt_rad, mrt_radius_m)
        bus_dist, bus_cnt = _nearest_distance_and_count(query, bus_rad, bus_radius_m)

        df.loc[has_coord, "dist_nearest_mrt_m"] = mrt_dist
        df.loc[has_coord, "dist_nearest_bus_m"] = bus_dist
        df.loc[has_coord, "n_mrt_1km"] = mrt_cnt
        df.loc[has_coord, "n_bus_400m"] = bus_cnt

    return df


def spatial_features_for_point(
    lat: float, lon: float, mrt_radius_m: float = 1000.0, bus_radius_m: float = 400.0
) -> dict:
    """Compute MRT/bus distance features for a single (lat, lon) point."""
    query = np.array([[lat, lon]])
    mrt_rad = _load_points("mrt_stations.csv")
    bus_rad = _load_points("bus_stops.csv")

    mrt_dist, mrt_cnt = _nearest_distance_and_count(query, mrt_rad, mrt_radius_m)
    bus_dist, bus_cnt = _nearest_distance_and_count(query, bus_rad, bus_radius_m)

    return {
        "dist_nearest_mrt_m": float(mrt_dist[0]),
        "dist_nearest_bus_m": float(bus_dist[0]),
        "n_mrt_1km": float(mrt_cnt[0]),
        "n_bus_400m": float(bus_cnt[0]),
    }


SPATIAL_FEATURES = [
    "dist_nearest_mrt_m",
    "dist_nearest_bus_m",
    "n_mrt_1km",
    "n_bus_400m",
]
