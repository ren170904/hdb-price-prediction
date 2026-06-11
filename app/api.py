"""FastAPI web app for HDB price valuation and deal finding.

Run:  uvicorn app.api:app --reload          (or `make webapp`)
Then open http://localhost:8000
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sg_hdb_price_analysis.features.engineering import (
    CBD_LAT,
    CBD_LON,
    FLAT_TYPE_ORDER,
    TIME_BASE_YEAR,
    haversine_m,
)
from sg_hdb_price_analysis.features.spatial import spatial_features_for_point

ROOT = Path(__file__).parents[1]
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Singapore HDB Price API", version="1.0")

# ── Data loaded once at startup ──────────────────────────────────────────────
_artifact = joblib.load(ROOT / "models" / "lgbm_model.pkl")

_raw = pd.read_csv(
    ROOT / "data" / "raw" / "hdb_resale_prices_all.csv",
    usecols=["month", "town", "flat_type", "flat_model", "storey_range",
             "block", "street_name", "resale_price", "floor_area_sqm",
             "lease_commence_date"],
    parse_dates=["month"],
)
_raw["address"] = _raw["block"].astype(str) + " " + _raw["street_name"]

_coords = pd.read_csv(ROOT / "data" / "external" / "address_coords.csv").dropna(subset=["lat"])
_coords = _coords.set_index("address")[["lat", "lon"]]

_mrt = pd.read_csv(ROOT / "data" / "external" / "mrt_stations.csv").dropna(subset=["lat"])
_bus = pd.read_csv(ROOT / "data" / "external" / "bus_stops.csv").dropna(subset=["lat"])

_valuations_path = ROOT / "data" / "processed" / "valuations.csv"
_valuations = (
    pd.read_csv(_valuations_path, parse_dates=["month"]) if _valuations_path.exists() else None
)
if _valuations is not None:
    _valuations["address"] = _valuations["block"].astype(str) + " " + _valuations["street_name"]

# Per-model "sample" stats so the UI can show what each flat model looks like in the data.
_model_stats = {}
for _name, _d in _raw.groupby("flat_model"):
    _model_stats[_name] = {
        "n": int(len(_d)),
        "share": round(len(_d) / len(_raw) * 100, 1),
        "median_area": int(_d["floor_area_sqm"].median()),
        "median_price": int(_d["resale_price"].median()),
        "era": f"{int(_d['lease_commence_date'].quantile(.1))}–{int(_d['lease_commence_date'].quantile(.9))}",
    }

_META = {
    "towns": sorted(_raw["town"].unique()),
    "flat_types": sorted(_raw["flat_type"].unique()),
    "flat_models": sorted(_raw["flat_model"].unique()),
    "storey_ranges": sorted(_raw["storey_range"].unique()),
    "data_through": _raw["month"].max().strftime("%Y-%m"),
    "n_transactions": int(len(_raw)),
    "model_metrics": {k: round(float(v), 3) for k, v in _artifact.get("test_metrics", {}).items()},
    "flat_model_stats": _model_stats,
}


class PredictRequest(BaseModel):
    town: str
    flat_type: str
    flat_model: str
    storey_range: str
    floor_area_sqm: float = Field(gt=10, lt=400)
    lease_commence_year: int = Field(ge=1960, le=2030)
    year: int = Field(ge=2017, le=2035)
    month: int = Field(ge=1, le=12, default=6)
    address: Optional[str] = None  # real address → exact spatial features


def _storey_mid(storey_range: str) -> float:
    parts = storey_range.split(" TO ")
    return (float(parts[0]) + float(parts[-1])) / 2


def _nearest_name(lat: float, lon: float, stations: pd.DataFrame) -> str:
    dists = haversine_m(lat, lon, stations["lat"].to_numpy(), stations["lon"].to_numpy())
    return str(stations["name"].iloc[dists.argmin()])


# ── API routes ───────────────────────────────────────────────────────────────
@app.get("/api/meta")
def meta():
    return _META


@app.get("/api/addresses")
def addresses(town: str):
    addrs = _raw.loc[_raw["town"] == town, "address"].unique()
    return sorted(a for a in addrs if a in _coords.index)


@app.post("/api/predict")
def predict(req: PredictRequest):
    features = _artifact["features"]
    cat_features = _artifact["cat_features"]
    medians = _artifact.get("medians", {})

    inputs = {
        "floor_area_sqm": req.floor_area_sqm,
        "storey_mid": _storey_mid(req.storey_range),
        "remaining_lease_exact": max(0, req.lease_commence_year + 99 - req.year),
        "flat_type_ord": FLAT_TYPE_ORDER.get(req.flat_type, 4),
        "time_index": (req.year - TIME_BASE_YEAR) * 12 + req.month,
        "flat_age": max(0, req.year - req.lease_commence_year),
        "town": req.town,
        "flat_model": req.flat_model,
    }

    location = None
    if req.address and req.address in _coords.index:
        lat, lon = (float(v) for v in _coords.loc[req.address])
        spatial = spatial_features_for_point(lat, lon)
        inputs.update(spatial)
        inputs["lat"], inputs["lon"] = lat, lon
        inputs["dist_cbd_m"] = float(haversine_m(lat, lon, CBD_LAT, CBD_LON))
        location = {
            "lat": lat, "lon": lon,
            "dist_nearest_mrt_m": round(spatial["dist_nearest_mrt_m"]),
            "dist_nearest_bus_m": round(spatial["dist_nearest_bus_m"]),
            "dist_cbd_km": round(inputs["dist_cbd_m"] / 1000, 1),
            "nearest_mrt_name": _nearest_name(lat, lon, _mrt),
            "nearest_bus_name": _nearest_name(lat, lon, _bus),
        }

    # Median fallback for anything the request couldn't supply (e.g. no address).
    for f in features:
        if f not in inputs and f in medians:
            inputs[f] = medians[f]

    X = pd.DataFrame([{f: inputs[f] for f in features}])
    X[cat_features] = _artifact["encoder"].transform(X[cat_features])
    pred = float(_artifact["model"].predict(X)[0])

    comps = _raw[
        (_raw["town"] == req.town)
        & (_raw["flat_type"] == req.flat_type)
        & (_raw["month"].dt.year >= req.year - 2)
    ]["resale_price"]

    return {
        "predicted_price": round(pred),
        "location": location,
        "comparables": {
            "n": int(len(comps)),
            "median": round(float(comps.median())) if len(comps) else None,
        },
    }


@app.get("/api/deals")
def deals(
    kind: str = "under",
    threshold: float = 10.0,
    town: Optional[str] = None,
    flat_type: Optional[str] = None,
    limit: int = 100,
):
    if _valuations is None:
        raise HTTPException(503, "valuations.csv not found — run notebook 05 first")

    v = _valuations
    if town:
        v = v[v["town"] == town]
    if flat_type:
        v = v[v["flat_type"] == flat_type]

    if kind == "under":
        v = v[v["gap_pct"] < -threshold].nsmallest(limit, "gap_pct")
    elif kind == "over":
        v = v[v["gap_pct"] > threshold].nlargest(limit, "gap_pct")
    else:
        v = v.reindex(v["gap_pct"].abs().sort_values(ascending=False).index).head(limit)

    out = v[["month", "address", "town", "flat_type", "storey_range", "floor_area_sqm",
             "lat", "lon", "actual", "predicted", "gap_pct"]].copy()
    out["month"] = out["month"].dt.strftime("%Y-%m")
    # object dtype first, otherwise None is coerced back to NaN in float columns
    out = out.astype(object).where(out.notna(), None)
    return {
        "total_scored": int(len(_valuations)),
        "deals": out.to_dict(orient="records"),
    }


# ── Frontend ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
