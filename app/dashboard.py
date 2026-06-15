"""Streamlit dashboard for HDB resale price analysis and prediction.

Five pages, selected from the sidebar:
  1. Price Trends      — market-wide trend, distribution, price-vs-area
  2. Town Comparison   — ranking + town x flat-type heatmap
  3. Map View          — per-block prices on an interactive map (+ MRT overlay)
  4. Price Prediction  — value a single flat with the trained model
  5. Deal Finder       — recent transactions flagged over/under model fair value

Run with:  streamlit run app/dashboard.py   (or `make dashboard`).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# All paths are resolved relative to the project root (one level above app/).
ROOT = Path(__file__).parents[1]
RAW_CSV = ROOT / "data" / "raw" / "hdb_resale_prices_all.csv"
COORDS_CSV = ROOT / "data" / "external" / "address_coords.csv"
MRT_CSV = ROOT / "data" / "external" / "mrt_stations.csv"
LGBM_PKL = ROOT / "models" / "lgbm_model.pkl"
RF_PKL = ROOT / "models" / "rf_model.pkl"

st.set_page_config(page_title="Singapore HDB Price Analysis", layout="wide", page_icon="🏠")


# ── Cached data loaders ───────────────────────────────────────────────────────
# @st.cache_data memoises the return value so the 232k-row CSV is parsed only
# once per session instead of on every widget interaction.
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    """Load the raw transactions and derive the columns the dashboard needs."""
    df = pd.read_csv(RAW_CSV, parse_dates=["month"])
    df["year"] = df["month"].dt.year
    df["month_of_year"] = df["month"].dt.month
    # The API returns everything as strings — coerce the numeric columns.
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"], errors="coerce")
    # Remaining lease = 99-year leasehold minus age at time of sale.
    df["remaining_lease_years"] = (df["lease_commence_date"] + 99 - df["year"]).clip(lower=0)
    # Build the address key used to join geocoded coordinates.
    df["address"] = df["block"].astype(str).str.strip() + " " + df["street_name"].astype(str).str.strip()

    # Merge geocoded lat/lon if the geocoding step has been run.
    if COORDS_CSV.exists():
        coords = pd.read_csv(COORDS_CSV)
        df = df.merge(coords[["address", "lat", "lon"]], on="address", how="left")
    return df


@st.cache_data
def load_coords() -> pd.DataFrame | None:
    """Address → lat/lon lookup table, or None if geocoding hasn't run yet."""
    if COORDS_CSV.exists():
        return pd.read_csv(COORDS_CSV)
    return None


@st.cache_data
def load_mrt() -> pd.DataFrame | None:
    """MRT/LRT station coordinates for the map overlay, or None if missing."""
    if MRT_CSV.exists():
        return pd.read_csv(MRT_CSV)
    return None


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: Path):
    """Load a trained model artifact. @st.cache_resource keeps the unpickled
    object in memory (it isn't a serialisable dataframe). Returns None if absent."""
    if not path.exists():
        return None
    return joblib.load(path)


def storey_midpoint(storey_range: str) -> float:
    """Convert a storey band like '07 TO 09' to its midpoint (8.0)."""
    parts = storey_range.split(" TO ")
    if len(parts) == 2:
        return (float(parts[0]) + float(parts[1])) / 2
    return float(parts[0])


# Ordinal encoding of flat type (size order), used as a model feature.
FLAT_TYPE_ORDER = {
    "1 ROOM": 1, "2 ROOM": 2, "3 ROOM": 3, "4 ROOM": 4,
    "5 ROOM": 5, "EXECUTIVE": 6, "MULTI-GENERATION": 7,
}

# ── Sidebar: page selection ──────────────────────────────────────────────────
st.sidebar.title("🏠 HDB Price Dashboard")
page = st.sidebar.radio(
    "Select a page",
    ["📊 Price Trends", "🗺️ Town Comparison", "🌏 Map View",
     "🤖 Price Prediction", "💰 Deal Finder"],
)

# Without the raw data nothing works — guide the user to fetch it first.
if not RAW_CSV.exists():
    st.error(
        "Data not found. Run this first:\n"
        "```bash\npython -m sg_hdb_price_analysis.data.fetch\n```"
    )
    st.stop()

df = load_data()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: Price Trends
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Price Trends":
    st.title("📊 HDB Resale Price Trends")

    # Filters: year range + which flat types to include.
    col1, col2 = st.columns(2)
    with col1:
        year_range = st.slider(
            "Period", int(df["year"].min()), int(df["year"].max()),
            (2010, int(df["year"].max()))
        )
    with col2:
        flat_types = st.multiselect(
            "Flat type",
            sorted(df["flat_type"].dropna().unique()),
            default=["3 ROOM", "4 ROOM", "5 ROOM"],
        )

    mask = (
        (df["year"] >= year_range[0])
        & (df["year"] <= year_range[1])
        & (df["flat_type"].isin(flat_types))
    )
    dff = df[mask]

    # Headline KPIs for the current selection.
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions", f"{len(dff):,}")
    k2.metric("Median price", f"S${dff['resale_price'].median():,.0f}")
    k3.metric("Mean price", f"S${dff['resale_price'].mean():,.0f}")
    k4.metric("Highest price", f"S${dff['resale_price'].max():,.0f}")

    st.divider()

    # Monthly median price, one line per flat type.
    monthly = (
        dff.groupby(["month", "flat_type"])["resale_price"]
        .median()
        .reset_index()
    )
    fig1 = px.line(
        monthly, x="month", y="resale_price", color="flat_type",
        title="Monthly median resale price (S$)",
        labels={"resale_price": "Median price (S$)", "month": "Month", "flat_type": "Flat type"},
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Price distribution (overlaid histograms per flat type).
    fig2 = px.histogram(
        dff, x="resale_price", color="flat_type", nbins=80,
        barmode="overlay", opacity=0.7,
        title="Price distribution",
        labels={"resale_price": "Resale price (S$)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Price vs floor area, with an OLS trend line. Sampled to keep the plot light.
    sample = dff.sample(min(5000, len(dff)), random_state=42)
    fig3 = px.scatter(
        sample, x="floor_area_sqm", y="resale_price", color="flat_type",
        opacity=0.5, trendline="ols",
        title="Floor area vs resale price",
        labels={"floor_area_sqm": "Floor area (sqm)", "resale_price": "Resale price (S$)"},
    )
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: Town Comparison
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Town Comparison":
    st.title("🗺️ HDB Price Comparison by Town")

    year_filter = st.slider(
        "Reference year", int(df["year"].min()), int(df["year"].max()), int(df["year"].max())
    )
    # Price-per-sqm controls for unit-size mix; raw price does not.
    metric_choice = st.radio("Metric", ["Median price", "Price per sqm"], horizontal=True)

    dff = df[df["year"] == year_filter]
    metric_col = "resale_price" if metric_choice == "Median price" else "price_per_sqm"
    label = "Median price (S$)" if metric_choice == "Median price" else "Price per sqm (S$/sqm)"

    # Aggregate the chosen metric per town and rank descending.
    town_stats = (
        dff.groupby("town")[metric_col]
        .agg(["median", "mean", "count"])
        .rename(columns={"median": "Median", "mean": "Mean", "count": "Transactions"})
        .sort_values("Median", ascending=False)
        .reset_index()
    )

    fig4 = px.bar(
        town_stats, x="town", y="Median",
        color="Median", color_continuous_scale="RdYlGn_r",
        title=f"{year_filter} — {label} by town",
        labels={"town": "Town", "Median": label},
    )
    fig4.update_xaxes(tickangle=45)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Town breakdown")
    st.dataframe(
        town_stats.style.format({"Median": "S${:,.0f}", "Mean": "S${:,.0f}", "Transactions": "{:,}"}),
        use_container_width=True,
    )

    # Median price for every (town, flat type) combination as a heatmap.
    st.subheader("Town × flat-type median heatmap")
    pivot = (
        dff.groupby(["town", "flat_type"])["resale_price"]
        .median()
        .unstack(fill_value=0)
    )
    fig5 = px.imshow(
        pivot / 1000, text_auto=".0f",
        labels=dict(x="Flat type", y="Town", color="Median (S$ thousands)"),
        aspect="auto", color_continuous_scale="Blues",
        title="Median resale price (S$ thousands)",
    )
    st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: Map View
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌏 Map View":
    st.title("🌏 HDB Price Map")

    # This page needs geocoded coordinates — bail out gracefully if absent.
    if "lat" not in df.columns or df["lat"].notna().sum() == 0:
        st.warning(
            "Coordinate data not found. Run geocoding first:\n"
            "```bash\npython -m sg_hdb_price_analysis.data.geocode\n```"
        )
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        map_year = st.slider(
            "Year", int(df["year"].min()), int(df["year"].max()), int(df["year"].max())
        )
    with c2:
        map_flat_types = st.multiselect(
            "Flat type", sorted(df["flat_type"].dropna().unique()),
            default=sorted(df["flat_type"].dropna().unique()),
        )
    with c3:
        color_metric = st.radio("Colour by", ["resale_price", "price_per_sqm"], horizontal=True)
        show_mrt = st.checkbox("Show MRT/LRT stations", value=True)

    mask = (
        (df["year"] == map_year)
        & (df["flat_type"].isin(map_flat_types))
        & df["lat"].notna()
    )
    dff = df[mask]

    # Aggregate by block so the map shows one marker per building, not per sale.
    agg = (
        dff.groupby(["address", "town", "lat", "lon"])
        .agg(
            median_price=("resale_price", "median"),
            median_psm=("price_per_sqm", "median"),
            n=("resale_price", "size"),
        )
        .reset_index()
    )
    color_col = "median_price" if color_metric == "resale_price" else "median_psm"
    color_label = "Median price (S$)" if color_metric == "resale_price" else "Price per sqm (S$/sqm)"

    st.caption(f"{map_year} | {len(agg):,} blocks | {len(dff):,} transactions")

    # Marker colour = price metric, marker size = number of transactions.
    fig_map = px.scatter_mapbox(
        agg, lat="lat", lon="lon",
        color=color_col, size="n",
        hover_name="address",
        hover_data={"town": True, "median_price": ":,.0f", "n": True, "lat": False, "lon": False},
        color_continuous_scale="Turbo", size_max=15, zoom=10.5,
        center={"lat": 1.355, "lon": 103.82},
        labels={color_col: color_label},
        height=650,
    )

    # Overlay MRT/LRT stations on top of the price markers.
    if show_mrt:
        mrt = load_mrt()
        if mrt is not None:
            fig_map.add_scattermapbox(
                lat=mrt["lat"], lon=mrt["lon"], mode="markers",
                marker=dict(size=7, color="black", symbol="rail"),
                text=mrt["name"], name="MRT/LRT",
                hoverinfo="text",
            )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Quantify the transit premium: price vs distance to the nearest MRT.
    if "town" in dff.columns:
        st.subheader("Distance to MRT vs price")
        from sg_hdb_price_analysis.features.spatial import _load_points, _nearest_distance_and_count

        coords = dff[["lat", "lon"]].to_numpy()
        mrt_rad = _load_points("mrt_stations.csv")
        # Nearest-MRT distance in metres for every transaction on screen.
        d, _ = _nearest_distance_and_count(coords, mrt_rad, 1000.0)
        plot_df = dff.copy()
        plot_df["dist_mrt_m"] = d
        sample = plot_df.sample(min(4000, len(plot_df)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="dist_mrt_m", y="resale_price", color="flat_type",
            opacity=0.4, trendline="lowess",
            labels={"dist_mrt_m": "Distance to nearest MRT (m)", "resale_price": "Resale price (S$)"},
            title="Nearest-MRT distance vs resale price",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: Price Prediction
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Price Prediction":
    st.title("🤖 HDB Price Prediction")

    lgbm_artifact = load_model(LGBM_PKL)
    rf_artifact = load_model(RF_PKL)

    # Need at least one trained model to make predictions.
    if lgbm_artifact is None and rf_artifact is None:
        st.warning(
            "No trained model found. Run this:\n"
            "```bash\npython -m sg_hdb_price_analysis.models.train\n```"
        )
        st.stop()

    # Offer whichever models are available on disk.
    model_name = st.selectbox(
        "Model",
        (["LightGBM"] if lgbm_artifact else []) + (["Random Forest"] if rf_artifact else []),
    )
    artifact = lgbm_artifact if model_name == "LightGBM" else rf_artifact

    st.subheader("Enter the flat's details")
    c1, c2, c3 = st.columns(3)

    towns = sorted(df["town"].dropna().unique())
    flat_types = sorted(df["flat_type"].dropna().unique())
    flat_models = sorted(df["flat_model"].dropna().unique())
    storey_ranges = sorted(df["storey_range"].dropna().unique())

    with c1:
        town = st.selectbox("Town", towns, index=towns.index("TAMPINES") if "TAMPINES" in towns else 0)
        flat_type = st.selectbox("Flat type", flat_types, index=flat_types.index("4 ROOM") if "4 ROOM" in flat_types else 0)
    with c2:
        flat_model = st.selectbox("Flat model", flat_models)
        storey_range = st.selectbox("Storey", storey_ranges)
    with c3:
        floor_area = st.number_input("Floor area (sqm)", min_value=20.0, max_value=300.0, value=93.0, step=1.0)
        lease_commence = st.number_input("Lease commence year", min_value=1960, max_value=2024, value=1990, step=1)

    # If the model uses spatial features, let the user pick a real address so we
    # can compute exact MRT/bus distances from its geocoded coordinates.
    spatial_used = any(c in artifact["features"] for c in
                       ["dist_nearest_mrt_m", "dist_nearest_bus_m", "n_mrt_1km", "n_bus_400m"])
    address = None
    if spatial_used:
        town_addresses = sorted(df.loc[df["town"] == town, "address"].dropna().unique())
        address = st.selectbox(
            "Address (used for MRT/bus distances)", town_addresses,
            help="Distances to the nearest MRT and bus stop are computed from this real address.",
        )

    pred_year = st.number_input("Prediction year", min_value=2000, max_value=2030, value=2024, step=1)

    if st.button("Predict price", type="primary"):
        from sg_hdb_price_analysis.features.engineering import (
            CBD_LAT, CBD_LON, TIME_BASE_YEAR, haversine_m,
        )

        # Build the same engineered features the model was trained on.
        storey_mid = storey_midpoint(storey_range)
        remaining = max(0, lease_commence + 99 - pred_year)
        flat_type_ord = FLAT_TYPE_ORDER.get(flat_type, 4)

        cat_features = artifact["cat_features"]
        all_features = artifact["features"]
        medians = artifact.get("medians", {})

        input_dict = {
            "floor_area_sqm": floor_area,
            "storey_mid": storey_mid,
            "remaining_lease_exact": remaining,
            "flat_type_ord": flat_type_ord,
            # time_index = months since 2017-01; assume mid-year (month 6).
            "time_index": (pred_year - TIME_BASE_YEAR) * 12 + 6,
            "flat_age": max(0, pred_year - lease_commence),
            "town": town,
            "flat_model": flat_model,
        }

        # Derive coordinate-based features from the chosen address.
        mrt_info = None
        if spatial_used:
            from sg_hdb_price_analysis.features.spatial import spatial_features_for_point

            coords = load_coords()
            row = coords[coords["address"] == address] if coords is not None else None
            if row is not None and len(row) and pd.notna(row.iloc[0]["lat"]):
                lat, lon = float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])
                feats = spatial_features_for_point(lat, lon)
                input_dict.update(feats)
                input_dict["lat"] = lat
                input_dict["lon"] = lon
                input_dict["dist_cbd_m"] = float(haversine_m(lat, lon, CBD_LAT, CBD_LON))
                mrt_info = feats
                mrt_info["dist_cbd_m"] = input_dict["dist_cbd_m"]

        # Fill any feature we couldn't supply with its training-set median
        # (e.g. spatial features when no address was selected).
        for f in all_features:
            if f not in input_dict and f in medians:
                input_dict[f] = medians[f]

        # Assemble the single-row feature frame in the exact training order,
        # encode the categoricals, and predict.
        X_input = pd.DataFrame([{k: input_dict[k] for k in all_features}])
        X_enc = X_input.copy()
        X_enc[cat_features] = artifact["encoder"].transform(X_enc[cat_features])

        pred = artifact["model"].predict(X_enc)[0]

        st.divider()
        st.metric(f"Predicted resale price ({model_name})", f"S${pred:,.0f}")

        # Show the location context that fed into the prediction.
        if mrt_info is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("Nearest MRT", f"{mrt_info['dist_nearest_mrt_m']:,.0f} m")
            m2.metric("Nearest bus stop", f"{mrt_info['dist_nearest_bus_m']:,.0f} m")
            m3.metric("To CBD (Raffles Place)", f"{mrt_info['dist_cbd_m'] / 1000:,.1f} km")

        # Sanity-check the estimate against real recent comparable sales.
        st.subheader("Comparable transactions")
        comp = df[
            (df["town"] == town)
            & (df["flat_type"] == flat_type)
            & (df["year"] >= pred_year - 2)
        ]
        if len(comp) > 0:
            fig6 = px.histogram(
                comp, x="resale_price", nbins=40,
                title=f"Recent sale prices for {town} / {flat_type}",
                labels={"resale_price": "Resale price (S$)"},
            )
            # Mark where the prediction falls within the comparable distribution.
            fig6.add_vline(x=pred, line_dash="dash", line_color="red",
                           annotation_text=f"Prediction: S${pred:,.0f}")
            st.plotly_chart(fig6, use_container_width=True)

            st.caption(f"{len(comp):,} comparable sales  |  median: S${comp['resale_price'].median():,.0f}")
        else:
            st.info("No comparable transactions found.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: Deal Finder
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💰 Deal Finder":
    st.title("💰 Deal Finder")

    # This page reads the leakage-free valuations exported by notebook 05.
    VALUATIONS_CSV = ROOT / "data" / "processed" / "valuations.csv"
    if not VALUATIONS_CSV.exists():
        st.warning(
            "Valuation data not found. Run notebook 05 first:\n"
            "```bash\njupyter nbconvert --execute --inplace "
            "notebooks/05_overpriced_underpriced_analysis.ipynb\n```"
        )
        st.stop()

    @st.cache_data(show_spinner="Loading valuations…")
    def load_valuations() -> pd.DataFrame:
        v = pd.read_csv(VALUATIONS_CSV, parse_dates=["month"])
        v["address"] = v["block"].astype(str) + " " + v["street_name"]
        return v

    vals = load_valuations()

    with st.expander("ℹ️ How it works — model-based fair value"):
        st.markdown(
            "A LightGBM model trained **only** on data older than the last 12 months predicts "
            "the *fair value* of each transaction in the last 12 months (training and scored data "
            "are kept separate to avoid leakage).\n\n"
            "**Gap = (actual price − predicted fair value) ÷ predicted fair value**\n"
            "- 🟢 Large negative gap → **underpriced** (sold below fair value)\n"
            "- 🔴 Large positive gap → **overpriced** (sold above fair value)\n\n"
            "⚠️ A large gap can also reflect things the data can't see — renovation, view, or an urgent sale."
        )

    # Optional filters; threshold defines what counts as a "deal".
    c1, c2, c3 = st.columns(3)
    with c1:
        deal_towns = st.multiselect("Town", sorted(vals["town"].unique()))
    with c2:
        deal_flat_types = st.multiselect("Flat type", sorted(vals["flat_type"].unique()))
    with c3:
        thresh = st.slider("Gap threshold (±%)", 5, 25, 10)

    dff = vals.copy()
    if deal_towns:
        dff = dff[dff["town"].isin(deal_towns)]
    if deal_flat_types:
        dff = dff[dff["flat_type"].isin(deal_flat_types)]

    under = dff[dff["gap_pct"] < -thresh]   # sold below fair value
    over = dff[dff["gap_pct"] > thresh]     # sold above fair value

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions (last 12 months)", f"{len(dff):,}")
    k2.metric("🟢 Underpriced", f"{len(under):,}", f"{len(under) / max(len(dff), 1):.1%}", delta_color="off")
    k3.metric("🔴 Overpriced", f"{len(over):,}", f"{len(over) / max(len(dff), 1):.1%}", delta_color="off")
    k4.metric("Median |gap|", f"{dff['gap_pct'].abs().median():.1f}%")

    st.divider()

    # Map the flagged deals (only those beyond the threshold, with coordinates).
    geo = dff.dropna(subset=["lat"])
    flagged = geo[geo["gap_pct"].abs() > thresh]
    if len(flagged):
        st.subheader(f"Transactions priced >±{thresh}% off fair value")
        fig_deals = px.scatter_mapbox(
            flagged, lat="lat", lon="lon",
            color="gap_pct", color_continuous_scale="RdYlGn_r",
            range_color=[-25, 25],
            hover_name="address",
            hover_data={"town": True, "flat_type": True, "actual": ":,.0f",
                        "predicted": ":,.0f", "gap_pct": ":.1f", "lat": False, "lon": False},
            zoom=10.5, center={"lat": 1.355, "lon": 103.82}, height=550,
            labels={"gap_pct": "Gap (%)", "actual": "Actual price", "predicted": "Fair value"},
        )
        fig_deals.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig_deals, use_container_width=True)
        st.caption("🟢 Green = underpriced (below fair value)  /  🔴 Red = overpriced (above fair value)")

    # Ranked tables of the biggest bargains and overpayments.
    show_cols = ["month", "address", "town", "flat_type", "storey_range",
                 "floor_area_sqm", "actual", "predicted", "gap_pct"]
    col_labels = {
        "month": "Month", "address": "Address", "town": "Town", "flat_type": "Type",
        "storey_range": "Storey", "floor_area_sqm": "Area (sqm)",
        "actual": "Sold for (S$)", "predicted": "Fair value (S$)", "gap_pct": "Gap (%)",
    }

    def deal_table(d: pd.DataFrame) -> pd.DataFrame:
        out = d[show_cols].rename(columns=col_labels).copy()
        out["Month"] = out["Month"].dt.strftime("%Y-%m")
        return out.reset_index(drop=True)

    tab1, tab2 = st.tabs([f"🟢 Top underpriced ({len(under):,})", f"🔴 Top overpriced ({len(over):,})"])
    with tab1:
        st.dataframe(
            deal_table(under.nsmallest(100, "gap_pct")).style.format(
                {"Sold for (S$)": "{:,.0f}", "Fair value (S$)": "{:,.0f}", "Gap (%)": "{:+.1f}"}
            ).background_gradient(subset=["Gap (%)"], cmap="Greens_r"),
            use_container_width=True, height=400,
        )
    with tab2:
        st.dataframe(
            deal_table(over.nlargest(100, "gap_pct")).style.format(
                {"Sold for (S$)": "{:,.0f}", "Fair value (S$)": "{:,.0f}", "Gap (%)": "{:+.1f}"}
            ).background_gradient(subset=["Gap (%)"], cmap="Reds"),
            use_container_width=True, height=400,
        )

    # Distribution of gaps — most transactions cluster near 0% (efficient market).
    st.subheader("Gap distribution")
    fig_hist = px.histogram(
        dff, x="gap_pct", nbins=80,
        labels={"gap_pct": "Gap (%)"},
        title="Gap between actual price and model fair value",
    )
    fig_hist.add_vline(x=-thresh, line_dash="dash", line_color="green",
                       annotation_text=f"underpriced < -{thresh}%")
    fig_hist.add_vline(x=thresh, line_dash="dash", line_color="red",
                       annotation_text=f"overpriced > +{thresh}%")
    st.plotly_chart(fig_hist, use_container_width=True)
