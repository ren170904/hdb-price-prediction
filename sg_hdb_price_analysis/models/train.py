"""Train and compare LightGBM vs RandomForest for HDB price prediction."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sg_hdb_price_analysis.features.engineering import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_COLS,
    COORD_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_features,
)

MODELS_DIR = Path(__file__).parents[2] / "models"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def load_data() -> pd.DataFrame:
    raw = pd.read_csv(
        Path(__file__).parents[2] / "data" / "raw" / "hdb_resale_prices_all.csv",
        parse_dates=["month"],
    )
    return build_features(raw)


def prepare_xy(df: pd.DataFrame):
    cat_features = [c for c in CATEGORICAL_COLS if c in df.columns]
    # Coordinate-derived features that are actually present and populated.
    coord = [c for c in COORD_FEATURES if c in df.columns and df[c].notna().any()]
    numeric = BASE_NUMERIC_FEATURES + coord
    features = numeric + cat_features

    # Require non-null base features + target; coord NaNs get median-imputed.
    df = df.dropna(subset=BASE_NUMERIC_FEATURES + cat_features + [TARGET])

    medians = {}
    for col in coord:
        medians[col] = float(df[col].median())
        df[col] = df[col].fillna(medians[col])

    X = df[features]
    y = df[TARGET]
    return X, y, cat_features, features, medians


def evaluate(name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n=== {name} ===")
    print(f"  MAE:  S${mae:,.0f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    return {"model": name, "MAE": mae, "R2": r2, "MAPE": mape}


def train_lightgbm(X_train, y_train, X_test, y_test, cat_features):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cat_features] = enc.fit_transform(X_train[cat_features])
    X_test[cat_features] = enc.transform(X_test[cat_features])

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return model, enc


def train_random_forest(X_train, y_train, X_test, y_test, cat_features):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cat_features] = enc.fit_transform(X_train[cat_features])
    X_test[cat_features] = enc.transform(X_test[cat_features])

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, enc


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    df = load_data()
    X, y, cat_features, features, medians = prepare_xy(df)
    print(f"Features ({len(features)}): {features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    results = []

    # LightGBM
    print("\nTraining LightGBM…")
    lgbm_model, lgbm_enc = train_lightgbm(X_train, y_train, X_test, y_test, cat_features)
    X_test_enc = X_test.copy()
    X_test_enc[cat_features] = lgbm_enc.transform(X_test_enc[cat_features])
    results.append(evaluate("LightGBM", y_test, lgbm_model.predict(X_test_enc)))
    joblib.dump({"model": lgbm_model, "encoder": lgbm_enc, "cat_features": cat_features,
                 "features": features, "medians": medians},
                MODELS_DIR / "lgbm_model.pkl")

    # Random Forest
    print("\nTraining Random Forest…")
    rf_model, rf_enc = train_random_forest(X_train, y_train, X_test, y_test, cat_features)
    X_test_enc_rf = X_test.copy()
    X_test_enc_rf[cat_features] = rf_enc.transform(X_test_enc_rf[cat_features])
    results.append(evaluate("Random Forest", y_test, rf_model.predict(X_test_enc_rf)))
    joblib.dump({"model": rf_model, "encoder": rf_enc, "cat_features": cat_features,
                 "features": features, "medians": medians},
                MODELS_DIR / "rf_model.pkl")

    # Feature importances (LightGBM)
    importances = pd.DataFrame({
        "feature": features,
        "importance": lgbm_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n=== LightGBM Feature Importances ===")
    print(importances.to_string(index=False))
    importances.to_csv(PROCESSED_DIR / "feature_importances.csv", index=False)

    # Summary
    summary = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    best = summary.loc[summary["MAE"].idxmin(), "model"]
    print(f"\nBest model: {best}")


if __name__ == "__main__":
    main()
