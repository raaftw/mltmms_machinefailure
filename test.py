"""
RQ1: Remaining Useful Life (RUL) prediction from daily sensor + metadata.

What this script does
- Loads your CSV.
- Builds a per-row target: RUL_days = (first_failure_date_for_ID - DATE).days
- Drops rows after the first failure for each ID (avoids post-failure leakage).
- Builds leak-safe rolling features per ID (uses ONLY past values via shift(1)).
- Trains an XGBoost regressor with proper preprocessing (one-hot for categoricals).
- Uses a TIME-BASED split (train on earlier dates, test on later dates).
- Reports MAE and RMSE (in days).

Requirements
- pandas, numpy, scikit-learn, xgboost
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

from xgboost import XGBRegressor


CSV_PATH = "equipment_failure_data_1.csv"


def add_leak_safe_rolling_features(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    sensor_cols: list[str],
    windows: tuple[int, ...] = (7, 14, 30),
) -> pd.DataFrame:
    """
    Adds rolling mean/std/slope and 1-day lag delta features using ONLY past data:
    - For day t, all rolling stats are computed from days < t (shift(1)).
    """
    def _slope_from_window(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        valid = ~np.isnan(values)
        if valid.sum() < 2:
            return np.nan

        y = values[valid]
        x = np.arange(len(y), dtype=float)
        x_mean = x.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return 0.0
        return float(np.sum((x - x_mean) * (y - y.mean())) / denom)

    df = df.sort_values([id_col, date_col]).copy()
    g = df.groupby(id_col, sort=False)

    for col in sensor_cols:
        # 1-day lag and delta (past-only)
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_delta1"] = df[col] - df[f"{col}_lag1"]

        # Rolling features from past-only values
        past = g[col].shift(1)
        for w in windows:
            df[f"{col}_roll{w}_mean"] = past.rolling(w, min_periods=max(2, w // 3)).mean().reset_index(level=0, drop=True)
            df[f"{col}_roll{w}_std"]  = past.rolling(w, min_periods=max(2, w // 3)).std().reset_index(level=0, drop=True)
            df[f"{col}_roll{w}_slope"] = past.rolling(w, min_periods=max(2, w // 3)).apply(_slope_from_window, raw=True).reset_index(level=0, drop=True)

    return df


def build_rul_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs RUL label per row based on FIRST failure per ID:
      failure_date(ID) = min DATE where EQUIPMENT_FAILURE==1
      RUL_days = (failure_date(ID) - DATE).days

    Drops:
    - IDs with no observed failure (in this dataset you likely have all failing IDs,
      but this keeps things safe).
    - Rows after first failure date (DATE > failure_date), to avoid leakage.
    """
    df = df.copy()

    # Parse dates
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["DATE"])

    # Ensure target is numeric 0/1
    df["EQUIPMENT_FAILURE"] = pd.to_numeric(df["EQUIPMENT_FAILURE"], errors="coerce").fillna(0).astype(int)

    # First failure date per ID
    failure_dates = (
        df.loc[df["EQUIPMENT_FAILURE"] == 1, ["ID", "DATE"]]
        .groupby("ID", as_index=False)["DATE"]
        .min()
        .rename(columns={"DATE": "FIRST_FAILURE_DATE"})
    )

    df = df.merge(failure_dates, on="ID", how="inner")  # keep only IDs with a failure

    # Keep only rows on/before first failure
    df = df[df["DATE"] <= df["FIRST_FAILURE_DATE"]].copy()

    # RUL in days (0 on the failure day)
    df["RUL_DAYS"] = (df["FIRST_FAILURE_DATE"] - df["DATE"]).dt.days.astype(int)

    return df


def time_based_split_3way(
    df: pd.DataFrame,
    date_col: str = "DATE",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    """
    Time-based 3-way split using date cutoffs at quantiles.

    Returns: train_df, val_df, test_df, train_cutoff, val_cutoff
    """
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be in (0, 1)")
    if not (0 < val_frac < 1):
        raise ValueError("val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    train_cutoff = df[date_col].quantile(train_frac)
    val_cutoff = df[date_col].quantile(train_frac + val_frac)

    train_df = df[df[date_col] <= train_cutoff].copy()
    val_df = df[(df[date_col] > train_cutoff) & (df[date_col] <= val_cutoff)].copy()
    test_df = df[df[date_col] > val_cutoff].copy()
    return train_df, val_df, test_df, train_cutoff, val_cutoff


def evaluate_event_metrics(y_true_rul: np.ndarray, y_pred_rul: np.ndarray, horizon: int) -> dict[str, float]:
    """
    Evaluates event-style metrics for: "will fail within `horizon` days?"
    Positive class is RUL <= horizon.
    """
    true_event = (y_true_rul <= horizon).astype(int)
    pred_event = (y_pred_rul <= horizon).astype(int)

    precision = precision_score(true_event, pred_event, zero_division=0)
    recall = recall_score(true_event, pred_event, zero_division=0)
    f1 = f1_score(true_event, pred_event, zero_division=0)

    # Convert RUL prediction to risk score (higher => more likely near failure)
    risk_score = -np.asarray(y_pred_rul, dtype=float)
    if np.unique(true_event).size < 2:
        pr_auc = np.nan
    else:
        pr_auc = average_precision_score(true_event, risk_score)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else np.nan,
        "support_pos": int(true_event.sum()),
        "pred_pos": int(pred_event.sum()),
    }


def main():
    df = pd.read_csv(CSV_PATH)

    # Basic sanity checks
    required_cols = {
        "ID", "DATE", "REGION_CLUSTER", "MAINTENANCE_VENDOR", "MANUFACTURER", "WELL_GROUP",
        "S5", "S8", "S13", "S15", "S16", "S17", "S18", "S19",
        "AGE_OF_EQUIPMENT", "EQUIPMENT_FAILURE",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {sorted(missing)}")

    # Build RUL dataset
    df = build_rul_dataset(df)

    # Define sensors and add leak-safe rolling features
    sensor_cols = ["S5", "S8", "S13", "S15", "S16", "S17", "S18", "S19"]
    df = add_leak_safe_rolling_features(
        df=df,
        id_col="ID",
        date_col="DATE",
        sensor_cols=sensor_cols,
        windows=(7, 14, 30),
    )

    # Drop columns that would leak or aren’t used as features
    # (We keep DATE only for splitting; remove it from the model inputs)
    leak_cols = ["EQUIPMENT_FAILURE", "FIRST_FAILURE_DATE"]
    for c in leak_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Time-based split (train / validation / test)
    train_df, val_df, test_df, train_cutoff, val_cutoff = time_based_split_3way(
        df,
        date_col="DATE",
        train_frac=0.7,
        val_frac=0.15,
    )

    # Target
    y_train = train_df["RUL_DAYS"].values
    y_val = val_df["RUL_DAYS"].values
    y_test = test_df["RUL_DAYS"].values

    # Features
    X_train = train_df.drop(columns=["RUL_DAYS", "DATE"])
    X_val = val_df.drop(columns=["RUL_DAYS", "DATE"])
    X_test = test_df.drop(columns=["RUL_DAYS", "DATE"])

    # Identify categorical vs numeric columns
    categorical_cols = ["REGION_CLUSTER", "MAINTENANCE_VENDOR", "MANUFACTURER"]
    # WELL_GROUP is numeric-ish but could be treated as categorical; keep numeric by default
    # If you want it categorical: add "WELL_GROUP" to categorical_cols
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    # Preprocessing
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
    )

    # Model (good baseline; tune later)
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Predict
    val_preds = pipe.predict(X_val)
    val_preds = np.clip(val_preds, 0, None)  # RUL can't be negative

    test_preds = pipe.predict(X_test)
    test_preds = np.clip(test_preds, 0, None)  # RUL can't be negative

    # Evaluate
    mae_val = mean_absolute_error(y_val, val_preds)
    rmse_val = np.sqrt(mean_squared_error(y_val, val_preds))

    mae_test = mean_absolute_error(y_test, test_preds)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))

    print("=== RQ1: RUL prediction ===")
    print(f"Rows total (pre-failure only): {len(df):,}")
    print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,} | Test rows: {len(test_df):,}")
    print(f"Time split cutoffs DATE: train_end={train_cutoff.date()} | val_end={val_cutoff.date()}")
    print(f"Validation MAE (days):  {mae_val:.3f}")
    print(f"Validation RMSE (days): {rmse_val:.3f}")
    print(f"Test MAE (days):        {mae_test:.3f}")
    print(f"Test RMSE (days):       {rmse_test:.3f}")

    # Optional: performance near failure (harder, more important)
    for horizon in [30, 14, 7]:
        mask = y_test <= horizon
        if mask.sum() > 0:
            mae_h = mean_absolute_error(y_test[mask], test_preds[mask])
            rmse_h = np.sqrt(mean_squared_error(y_test[mask], test_preds[mask]))
            print(f"MAE/RMSE for last {horizon} days before failure (n={mask.sum()}): {mae_h:.3f} / {rmse_h:.3f}")

    print("\nEvent metrics on TEST (positive = failure within horizon):")
    for horizon in [30, 14, 7]:
        m = evaluate_event_metrics(y_test, test_preds, horizon)
        pr_auc_str = f"{m['pr_auc']:.3f}" if not np.isnan(m["pr_auc"]) else "NA"
        print(
            f"H={horizon:>2}d | "
            f"Precision={m['precision']:.3f} "
            f"Recall={m['recall']:.3f} "
            f"F1={m['f1']:.3f} "
            f"PR-AUC={pr_auc_str} "
            f"(true_pos={m['support_pos']}, pred_pos={m['pred_pos']})"
        )

    # Example: show a few predictions
    out = test_df[["ID", "DATE", "RUL_DAYS"]].copy()
    out["RUL_PRED"] = test_preds
    print("\nSample predictions:")
    print(out.sort_values(["ID", "DATE"]).head(10).to_string(index=False))


if __name__ == "__main__":
    main()