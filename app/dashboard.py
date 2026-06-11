"""Streamlit dashboard for HDB resale price analysis and prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parents[1]
RAW_CSV = ROOT / "data" / "raw" / "hdb_resale_prices_all.csv"
COORDS_CSV = ROOT / "data" / "external" / "address_coords.csv"
MRT_CSV = ROOT / "data" / "external" / "mrt_stations.csv"
LGBM_PKL = ROOT / "models" / "lgbm_model.pkl"
RF_PKL = ROOT / "models" / "rf_model.pkl"

st.set_page_config(page_title="Singapore HDB Price Analysis", layout="wide", page_icon="🏠")


@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, parse_dates=["month"])
    df["year"] = df["month"].dt.year
    df["month_of_year"] = df["month"].dt.month
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"], errors="coerce")
    df["remaining_lease_years"] = (df["lease_commence_date"] + 99 - df["year"]).clip(lower=0)
    df["address"] = df["block"].astype(str).str.strip() + " " + df["street_name"].astype(str).str.strip()

    # Merge geocoded coordinates if available
    if COORDS_CSV.exists():
        coords = pd.read_csv(COORDS_CSV)
        df = df.merge(coords[["address", "lat", "lon"]], on="address", how="left")
    return df


@st.cache_data
def load_coords() -> pd.DataFrame | None:
    if COORDS_CSV.exists():
        return pd.read_csv(COORDS_CSV)
    return None


@st.cache_data
def load_mrt() -> pd.DataFrame | None:
    if MRT_CSV.exists():
        return pd.read_csv(MRT_CSV)
    return None


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def storey_midpoint(storey_range: str) -> float:
    parts = storey_range.split(" TO ")
    if len(parts) == 2:
        return (float(parts[0]) + float(parts[1])) / 2
    return float(parts[0])


FLAT_TYPE_ORDER = {
    "1 ROOM": 1, "2 ROOM": 2, "3 ROOM": 3, "4 ROOM": 4,
    "5 ROOM": 5, "EXECUTIVE": 6, "MULTI-GENERATION": 7,
}

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🏠 HDB Price Dashboard")
page = st.sidebar.radio(
    "ページ選択",
    ["📊 価格トレンド分析", "🗺️ 地域別比較", "🌏 地図可視化", "🤖 価格予測", "💰 割安・割高物件検索"],
)

data_loaded = RAW_CSV.exists()

if not data_loaded:
    st.error(
        "データが見つかりません。まず以下を実行してください:\n"
        "```bash\npython -m sg_hdb_price_analysis.data.fetch\n```"
    )
    st.stop()

df = load_data()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: 価格トレンド分析
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 価格トレンド分析":
    st.title("📊 HDB Resale Price トレンド分析")

    col1, col2 = st.columns(2)
    with col1:
        year_range = st.slider(
            "期間", int(df["year"].min()), int(df["year"].max()),
            (2010, int(df["year"].max()))
        )
    with col2:
        flat_types = st.multiselect(
            "フラットタイプ",
            sorted(df["flat_type"].dropna().unique()),
            default=["3 ROOM", "4 ROOM", "5 ROOM"],
        )

    mask = (
        (df["year"] >= year_range[0])
        & (df["year"] <= year_range[1])
        & (df["flat_type"].isin(flat_types))
    )
    dff = df[mask]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("取引件数", f"{len(dff):,}")
    k2.metric("中央値価格", f"S${dff['resale_price'].median():,.0f}")
    k3.metric("平均価格", f"S${dff['resale_price'].mean():,.0f}")
    k4.metric("最高価格", f"S${dff['resale_price'].max():,.0f}")

    st.divider()

    # Monthly trend
    monthly = (
        dff.groupby(["month", "flat_type"])["resale_price"]
        .median()
        .reset_index()
    )
    fig1 = px.line(
        monthly, x="month", y="resale_price", color="flat_type",
        title="月次中央値成約価格 (S$)",
        labels={"resale_price": "中央値価格 (S$)", "month": "月", "flat_type": "フラットタイプ"},
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Price distribution
    fig2 = px.histogram(
        dff, x="resale_price", color="flat_type", nbins=80,
        barmode="overlay", opacity=0.7,
        title="価格分布",
        labels={"resale_price": "成約価格 (S$)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Price vs floor area scatter
    sample = dff.sample(min(5000, len(dff)), random_state=42)
    fig3 = px.scatter(
        sample, x="floor_area_sqm", y="resale_price", color="flat_type",
        opacity=0.5, trendline="ols",
        title="床面積 vs 成約価格",
        labels={"floor_area_sqm": "床面積 (㎡)", "resale_price": "成約価格 (S$)"},
    )
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: 地域別比較
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🗺️ 地域別比較":
    st.title("🗺️ 地域別 HDB 価格比較")

    year_filter = st.slider(
        "基準年", int(df["year"].min()), int(df["year"].max()), int(df["year"].max())
    )
    metric_choice = st.radio("指標", ["中央値価格", "㎡単価"], horizontal=True)

    dff = df[df["year"] == year_filter]
    metric_col = "resale_price" if metric_choice == "中央値価格" else "price_per_sqm"
    label = "中央値価格 (S$)" if metric_choice == "中央値価格" else "㎡単価 (S$/㎡)"

    town_stats = (
        dff.groupby("town")[metric_col]
        .agg(["median", "mean", "count"])
        .rename(columns={"median": "中央値", "mean": "平均", "count": "取引件数"})
        .sort_values("中央値", ascending=False)
        .reset_index()
    )

    fig4 = px.bar(
        town_stats, x="town", y="中央値",
        color="中央値", color_continuous_scale="RdYlGn_r",
        title=f"{year_filter}年 地域別{label}",
        labels={"town": "地域", "中央値": label},
    )
    fig4.update_xaxes(tickangle=45)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("地域別詳細")
    st.dataframe(
        town_stats.style.format({"中央値": "S${:,.0f}", "平均": "S${:,.0f}", "取引件数": "{:,}"}),
        use_container_width=True,
    )

    # Flat type heatmap by town
    st.subheader("地域 × フラットタイプ 中央値ヒートマップ")
    pivot = (
        dff.groupby(["town", "flat_type"])["resale_price"]
        .median()
        .unstack(fill_value=0)
    )
    fig5 = px.imshow(
        pivot / 1000, text_auto=".0f",
        labels=dict(x="フラットタイプ", y="地域", color="中央値 (S$千)"),
        aspect="auto", color_continuous_scale="Blues",
        title="中央値成約価格 (S$千)",
    )
    st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: 地図可視化
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌏 地図可視化":
    st.title("🌏 HDB 価格 地図可視化")

    if "lat" not in df.columns or df["lat"].notna().sum() == 0:
        st.warning(
            "座標データが見つかりません。先にジオコーディングを実行してください:\n"
            "```bash\npython -m sg_hdb_price_analysis.data.geocode\n```"
        )
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        map_year = st.slider(
            "年", int(df["year"].min()), int(df["year"].max()), int(df["year"].max())
        )
    with c2:
        map_flat_types = st.multiselect(
            "フラットタイプ", sorted(df["flat_type"].dropna().unique()),
            default=sorted(df["flat_type"].dropna().unique()),
        )
    with c3:
        color_metric = st.radio("色分け指標", ["resale_price", "price_per_sqm"], horizontal=True)
        show_mrt = st.checkbox("MRT/LRT駅を表示", value=True)

    mask = (
        (df["year"] == map_year)
        & (df["flat_type"].isin(map_flat_types))
        & df["lat"].notna()
    )
    dff = df[mask]

    # Aggregate by address to keep the map light and meaningful
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
    color_label = "中央値価格 (S$)" if color_metric == "resale_price" else "㎡単価 (S$/㎡)"

    st.caption(f"{map_year}年 | {len(agg):,} 棟 | {len(dff):,} 取引")

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

    # MRT distance vs price relationship
    if "town" in dff.columns:
        st.subheader("MRTからの距離と価格の関係")
        from sg_hdb_price_analysis.features.spatial import _load_points, _nearest_distance_and_count
        import numpy as _np

        coords = dff[["lat", "lon"]].to_numpy()
        mrt_rad = _load_points("mrt_stations.csv")
        d, _ = _nearest_distance_and_count(coords, mrt_rad, 1000.0)
        plot_df = dff.copy()
        plot_df["dist_mrt_m"] = d
        sample = plot_df.sample(min(4000, len(plot_df)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="dist_mrt_m", y="resale_price", color="flat_type",
            opacity=0.4, trendline="lowess",
            labels={"dist_mrt_m": "最寄りMRTまでの距離 (m)", "resale_price": "成約価格 (S$)"},
            title="最寄りMRT距離 vs 成約価格",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: 価格予測
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 価格予測":
    st.title("🤖 HDB 価格予測")

    lgbm_artifact = load_model(LGBM_PKL)
    rf_artifact = load_model(RF_PKL)

    if lgbm_artifact is None and rf_artifact is None:
        st.warning(
            "学習済みモデルが見つかりません。以下を実行してください:\n"
            "```bash\npython -m sg_hdb_price_analysis.models.train\n```"
        )
        st.stop()

    model_name = st.selectbox(
        "モデル選択",
        (["LightGBM"] if lgbm_artifact else []) + (["Random Forest"] if rf_artifact else []),
    )
    artifact = lgbm_artifact if model_name == "LightGBM" else rf_artifact

    st.subheader("物件情報を入力してください")
    c1, c2, c3 = st.columns(3)

    towns = sorted(df["town"].dropna().unique())
    flat_types = sorted(df["flat_type"].dropna().unique())
    flat_models = sorted(df["flat_model"].dropna().unique())
    storey_ranges = sorted(df["storey_range"].dropna().unique())

    with c1:
        town = st.selectbox("地域 (Town)", towns, index=towns.index("TAMPINES") if "TAMPINES" in towns else 0)
        flat_type = st.selectbox("フラットタイプ", flat_types, index=flat_types.index("4 ROOM") if "4 ROOM" in flat_types else 0)
    with c2:
        flat_model = st.selectbox("フラットモデル", flat_models)
        storey_range = st.selectbox("階数", storey_ranges)
    with c3:
        floor_area = st.number_input("床面積 (㎡)", min_value=20.0, max_value=300.0, value=93.0, step=1.0)
        lease_commence = st.number_input("リース開始年", min_value=1960, max_value=2024, value=1990, step=1)

    # Optional: pick a real address in the selected town to derive spatial features
    spatial_used = any(c in artifact["features"] for c in
                       ["dist_nearest_mrt_m", "dist_nearest_bus_m", "n_mrt_1km", "n_bus_400m"])
    address = None
    if spatial_used:
        town_addresses = sorted(df.loc[df["town"] == town, "address"].dropna().unique())
        address = st.selectbox(
            "住所 (MRT/バス停距離の計算に使用)", town_addresses,
            help="選択した実在の住所の座標から最寄りMRT・バス停までの距離を計算します。",
        )

    pred_year = st.number_input("予測年", min_value=2000, max_value=2030, value=2024, step=1)

    if st.button("価格を予測する", type="primary"):
        from sg_hdb_price_analysis.features.engineering import (
            CBD_LAT, CBD_LON, TIME_BASE_YEAR, haversine_m,
        )

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
            "time_index": (pred_year - TIME_BASE_YEAR) * 12 + 6,
            "flat_age": max(0, pred_year - lease_commence),
            "town": town,
            "flat_model": flat_model,
        }

        # Coordinate-derived features from the selected address
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

        # Any still-missing feature → median fallback
        for f in all_features:
            if f not in input_dict and f in medians:
                input_dict[f] = medians[f]

        X_input = pd.DataFrame([{k: input_dict[k] for k in all_features}])
        X_enc = X_input.copy()
        X_enc[cat_features] = artifact["encoder"].transform(X_enc[cat_features])

        pred = artifact["model"].predict(X_enc)[0]

        st.divider()
        st.metric(f"予測成約価格 ({model_name})", f"S${pred:,.0f}")

        if mrt_info is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("最寄りMRTまで", f"{mrt_info['dist_nearest_mrt_m']:,.0f} m")
            m2.metric("最寄りバス停まで", f"{mrt_info['dist_nearest_bus_m']:,.0f} m")
            m3.metric("CBD(ラッフルズ)まで", f"{mrt_info['dist_cbd_m'] / 1000:,.1f} km")

        # Show comparable transactions
        st.subheader("類似物件の実績")
        comp = df[
            (df["town"] == town)
            & (df["flat_type"] == flat_type)
            & (df["year"] >= pred_year - 2)
        ]
        if len(comp) > 0:
            fig6 = px.histogram(
                comp, x="resale_price", nbins=40,
                title=f"{town} / {flat_type} の直近成約価格分布",
                labels={"resale_price": "成約価格 (S$)"},
            )
            fig6.add_vline(x=pred, line_dash="dash", line_color="red",
                           annotation_text=f"予測: S${pred:,.0f}")
            st.plotly_chart(fig6, use_container_width=True)

            st.caption(f"類似物件 {len(comp):,} 件  |  中央値: S${comp['resale_price'].median():,.0f}")
        else:
            st.info("類似物件データが見つかりませんでした。")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: 割安・割高物件検索 (Deal Finder)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💰 割安・割高物件検索":
    st.title("💰 割安・割高物件検索 (Deal Finder)")

    VALUATIONS_CSV = ROOT / "data" / "processed" / "valuations.csv"
    if not VALUATIONS_CSV.exists():
        st.warning(
            "評価データが見つかりません。先にノートブック5を実行してください:\n"
            "```bash\njupyter nbconvert --execute --inplace "
            "notebooks/05_overpriced_underpriced_analysis.ipynb\n```"
        )
        st.stop()

    @st.cache_data(show_spinner="評価データを読み込み中…")
    def load_valuations() -> pd.DataFrame:
        v = pd.read_csv(VALUATIONS_CSV, parse_dates=["month"])
        v["address"] = v["block"].astype(str) + " " + v["street_name"]
        return v

    vals = load_valuations()

    with st.expander("ℹ️ 仕組み — モデルによる適正価格評価"):
        st.markdown(
            "直近12ヶ月**より前**のデータのみで学習したLightGBMモデルが、直近12ヶ月の各取引の"
            "「適正価格」を予測します(学習データと評価対象を分離しリーケージを防止)。\n\n"
            "**乖離率 (gap) = (実際の成約価格 − 予測適正価格) ÷ 予測適正価格**\n"
            "- 🟢 乖離率が大きくマイナス → **割安**(適正価格より安く成約)\n"
            "- 🔴 乖離率が大きくプラス → **割高**(適正価格より高く成約)\n\n"
            "⚠️ 大きな乖離はリノベーション・眺望・急売りなどデータに表れない要因の可能性もあります。"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        deal_towns = st.multiselect("地域", sorted(vals["town"].unique()))
    with c2:
        deal_flat_types = st.multiselect("フラットタイプ", sorted(vals["flat_type"].unique()))
    with c3:
        thresh = st.slider("乖離率しきい値 (±%)", 5, 25, 10)

    dff = vals.copy()
    if deal_towns:
        dff = dff[dff["town"].isin(deal_towns)]
    if deal_flat_types:
        dff = dff[dff["flat_type"].isin(deal_flat_types)]

    under = dff[dff["gap_pct"] < -thresh]
    over = dff[dff["gap_pct"] > thresh]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("対象取引 (直近12ヶ月)", f"{len(dff):,}")
    k2.metric("🟢 割安物件", f"{len(under):,}", f"{len(under) / max(len(dff), 1):.1%}", delta_color="off")
    k3.metric("🔴 割高物件", f"{len(over):,}", f"{len(over) / max(len(dff), 1):.1%}", delta_color="off")
    k4.metric("乖離率の中央値(絶対値)", f"{dff['gap_pct'].abs().median():.1f}%")

    st.divider()

    # Map of flagged deals
    geo = dff.dropna(subset=["lat"])
    flagged = geo[geo["gap_pct"].abs() > thresh]
    if len(flagged):
        st.subheader(f"乖離率 ±{thresh}% 超の物件マップ")
        fig_deals = px.scatter_mapbox(
            flagged, lat="lat", lon="lon",
            color="gap_pct", color_continuous_scale="RdYlGn_r",
            range_color=[-25, 25],
            hover_name="address",
            hover_data={"town": True, "flat_type": True, "actual": ":,.0f",
                        "predicted": ":,.0f", "gap_pct": ":.1f", "lat": False, "lon": False},
            zoom=10.5, center={"lat": 1.355, "lon": 103.82}, height=550,
            labels={"gap_pct": "乖離率 (%)", "actual": "成約価格", "predicted": "適正価格"},
        )
        fig_deals.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig_deals, use_container_width=True)
        st.caption("🟢 緑 = 割安(適正価格より安い) / 🔴 赤 = 割高(適正価格より高い)")

    # Tables
    show_cols = ["month", "address", "town", "flat_type", "storey_range",
                 "floor_area_sqm", "actual", "predicted", "gap_pct"]
    col_labels = {
        "month": "取引月", "address": "住所", "town": "地域", "flat_type": "タイプ",
        "storey_range": "階数", "floor_area_sqm": "面積(㎡)",
        "actual": "成約価格(S$)", "predicted": "適正価格(S$)", "gap_pct": "乖離率(%)",
    }

    def deal_table(d: pd.DataFrame) -> pd.DataFrame:
        out = d[show_cols].rename(columns=col_labels).copy()
        out["取引月"] = out["取引月"].dt.strftime("%Y-%m")
        return out.reset_index(drop=True)

    tab1, tab2 = st.tabs([f"🟢 割安トップ ({len(under):,}件)", f"🔴 割高トップ ({len(over):,}件)"])
    with tab1:
        st.dataframe(
            deal_table(under.nsmallest(100, "gap_pct")).style.format(
                {"成約価格(S$)": "{:,.0f}", "適正価格(S$)": "{:,.0f}", "乖離率(%)": "{:+.1f}"}
            ).background_gradient(subset=["乖離率(%)"], cmap="Greens_r"),
            use_container_width=True, height=400,
        )
    with tab2:
        st.dataframe(
            deal_table(over.nlargest(100, "gap_pct")).style.format(
                {"成約価格(S$)": "{:,.0f}", "適正価格(S$)": "{:,.0f}", "乖離率(%)": "{:+.1f}"}
            ).background_gradient(subset=["乖離率(%)"], cmap="Reds"),
            use_container_width=True, height=400,
        )

    # Gap distribution
    st.subheader("乖離率の分布")
    fig_hist = px.histogram(
        dff, x="gap_pct", nbins=80,
        labels={"gap_pct": "乖離率 (%)"},
        title="実際の成約価格とモデル適正価格の乖離率",
    )
    fig_hist.add_vline(x=-thresh, line_dash="dash", line_color="green",
                       annotation_text=f"割安 < -{thresh}%")
    fig_hist.add_vline(x=thresh, line_dash="dash", line_color="red",
                       annotation_text=f"割高 > +{thresh}%")
    st.plotly_chart(fig_hist, use_container_width=True)
