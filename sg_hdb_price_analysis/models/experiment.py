"""Feature experiment harness: test feature-set variants and measure LightGBM lift."""

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

# CBD reference point: Raffles Place
CBD_LAT, CBD_LON = 1.2830, 103.8513
EARTH_R = 6_371_000.0

MATURE_ESTATES = {
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL AREA",
    "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS",
    "QUEENSTOWN", "SERANGOON", "TAMPINES", "TOA PAYOH",
}

REGION_MAP = {
    "ANG MO KIO": "NE", "HOUGANG": "NE", "PUNGGOL": "NE", "SENGKANG": "NE",
    "SERANGOON": "NE",
    "BEDOK": "E", "PASIR RIS": "E", "TAMPINES": "E",
    "BISHAN": "C", "BUKIT MERAH": "C", "BUKIT TIMAH": "C", "CENTRAL AREA": "C",
    "GEYLANG": "C", "KALLANG/WHAMPOA": "C", "MARINE PARADE": "C", "QUEENSTOWN": "C",
    "TOA PAYOH": "C",
    "BUKIT BATOK": "W", "BUKIT PANJANG": "W", "CHOA CHU KANG": "W", "CLEMENTI": "W",
    "JURONG EAST": "W", "JURONG WEST": "W",
    "SEMBAWANG": "N", "WOODLANDS": "N", "YISHUN": "N",
}


def parse_remaining_lease(s) -> float:
    """Parse 'NN years MM months' → float years. Falls back to NaN."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    years, months = 0.0, 0.0
    parts = s.split()
    for i, tok in enumerate(parts):
        if tok.startswith("year"):
            years = float(parts[i - 1])
        elif tok.startswith("month"):
            months = float(parts[i - 1])
    return years + months / 12.0


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add all candidate features on top of the cached feature frame."""
    df = df.copy()

    # Accurate remaining lease (vs estimated)
    df["remaining_lease_exact"] = df["remaining_lease"].apply(parse_remaining_lease)
    # Fallback to estimate where missing
    df["remaining_lease_exact"] = df["remaining_lease_exact"].fillna(df["remaining_lease_years"])

    # Flat age at time of sale
    df["flat_age"] = (df["year"] - df["lease_commence_date"]).clip(lower=0)

    # Continuous time index (months since 2017-01)
    df["time_index"] = (df["year"] - 2017) * 12 + df["month_of_year"]

    # Distance to CBD (haversine, metres)
    lat1, lon1 = np.radians(df["lat"]), np.radians(df["lon"])
    lat2, lon2 = np.radians(CBD_LAT), np.radians(CBD_LON)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["dist_cbd_m"] = 2 * EARTH_R * np.arcsin(np.sqrt(a))

    # Estate maturity & region
    df["is_mature"] = df["town"].isin(MATURE_ESTATES).astype(int)
    df["region"] = df["town"].map(REGION_MAP).fillna("C")

    # Area per "room" proxy (interaction)
    df["area_per_lease"] = df["floor_area_sqm"] / (df["remaining_lease_exact"] + 1)

    return df


def run(df: pd.DataFrame, numeric, categorical, name: str, seed: int = 42) -> dict:
    feats = numeric + categorical
    d = df.dropna(subset=[c for c in feats if c not in categorical] + ["resale_price"]).copy()
    X = d[feats].copy()
    y = d["resale_price"]

    # Median-impute any remaining numeric NaN (e.g. spatial for ungeocoded)
    for c in numeric:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    if categorical:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr[categorical] = enc.fit_transform(Xtr[categorical])
        Xte[categorical] = enc.transform(Xte[categorical])

    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    r2 = r2_score(yte, pred)
    mape = np.mean(np.abs((yte - pred) / yte)) * 100
    print(f"{name:42s}  MAE S${mae:>7,.0f}  R² {r2:.4f}  MAPE {mape:.3f}%  ({len(feats)} feat)")
    return {"name": name, "MAE": mae, "R2": r2, "MAPE": mape, "n_feat": len(feats)}


if __name__ == "__main__":
    df = pd.read_parquet("data/interim/features_full.parquet")
    df = engineer(df)

    BASE_NUM = ["floor_area_sqm", "storey_mid", "remaining_lease_years", "flat_type_ord",
                "year", "month_of_year"]
    SPATIAL = ["dist_nearest_mrt_m", "dist_nearest_bus_m", "n_mrt_1km", "n_bus_400m"]
    BASE_CAT = ["town", "flat_type", "flat_model", "storey_range"]

    print("=" * 90)
    # A. Current production model
    run(df, BASE_NUM + SPATIAL, BASE_CAT, "A. Current (production)")

    # B. Drop zero-importance redundant cats
    run(df, BASE_NUM + SPATIAL, ["town", "flat_model"], "B. Drop flat_type+storey_range cats")

    # C. + exact remaining lease (replace estimate)
    num_c = ["floor_area_sqm", "storey_mid", "remaining_lease_exact", "flat_type_ord",
             "year", "month_of_year"] + SPATIAL
    run(df, num_c, ["town", "flat_model"], "C. B + exact remaining_lease")

    # D. + flat_age, dist_cbd
    run(df, num_c + ["flat_age", "dist_cbd_m"], ["town", "flat_model"],
        "D. C + flat_age + dist_cbd")

    # E. + raw coordinates
    run(df, num_c + ["flat_age", "dist_cbd_m", "lat", "lon"], ["town", "flat_model"],
        "E. D + lat/lon")

    # F. + time_index (replace year+month), is_mature, region
    num_f = ["floor_area_sqm", "storey_mid", "remaining_lease_exact", "flat_type_ord",
             "time_index"] + SPATIAL + ["flat_age", "dist_cbd_m", "lat", "lon", "is_mature"]
    run(df, num_f, ["town", "flat_model", "region"], "F. E + time_index + is_mature + region")

    # G. F minus weak (flat_type_ord, n_mrt_1km) test
    num_g = ["floor_area_sqm", "storey_mid", "remaining_lease_exact", "time_index",
             "dist_nearest_mrt_m", "dist_nearest_bus_m", "n_bus_400m",
             "flat_age", "dist_cbd_m", "lat", "lon", "is_mature"]
    run(df, num_g, ["town", "flat_model"], "G. F trimmed (drop weak feats)")
    print("=" * 90)
