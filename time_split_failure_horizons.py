"""
Time-split multiclass failure-horizon prediction.

Classes:
- 0: NOW        (RUL_DAYS == 0)
- 1: D1_7       (1 <= RUL_DAYS <= 7)
- 2: D8_30      (8 <= RUL_DAYS <= 30)
- 3: GT_30      (RUL_DAYS > 30)

This script:
- Builds RUL safely from first failure per ID
- Uses strict time-based train/val/test split
- Trains one multiclass model (LogisticRegression)
- Reports class distribution and multiclass metrics
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CSV_PATH = "equipment_failure_data_1.csv"
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_MAX_TRAIN_ROWS = 0
DEFAULT_N_ESTIMATORS = 400

# Cost-sensitive weighting: higher weights push the model to care more about near-failure classes.
CLASS_WEIGHT_NEAR_FAILURE = {0: 20.0, 1: 12.0, 2: 6.0, 3: 1.0}

CLASS_NAMES = {
    0: "NOW",
    1: "D1_7",
    2: "D8_30",
    3: "GT_30",
}


@dataclass
class Result:
    name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    close_recall_macro: float


def build_rul_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DATE"] = pd.to_datetime(out["DATE"], format="%m/%d/%y", errors="coerce")
    out["EQUIPMENT_FAILURE"] = pd.to_numeric(out["EQUIPMENT_FAILURE"], errors="coerce").fillna(0).astype(int)
    out = out.dropna(subset=["DATE"])

    first_failure = (
        out.loc[out["EQUIPMENT_FAILURE"] == 1, ["ID", "DATE"]]
        .groupby("ID", as_index=False)["DATE"]
        .min()
        .rename(columns={"DATE": "FIRST_FAILURE_DATE"})
    )

    out = out.merge(first_failure, on="ID", how="inner")
    out = out[out["DATE"] <= out["FIRST_FAILURE_DATE"]].copy()
    out["RUL_DAYS"] = (out["FIRST_FAILURE_DATE"] - out["DATE"]).dt.days.astype(int)
    return out.sort_values(["ID", "DATE"]).reset_index(drop=True)


def add_horizon_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["FAILURE_HORIZON_CLASS"] = np.select(
        [
            out["RUL_DAYS"] == 0,
            (out["RUL_DAYS"] >= 1) & (out["RUL_DAYS"] <= 7),
            (out["RUL_DAYS"] >= 8) & (out["RUL_DAYS"] <= 30),
            out["RUL_DAYS"] > 30,
        ],
        [0, 1, 2, 3],
        default=3,
    ).astype(int)
    return out


def add_leak_safe_features(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    out = df.sort_values(["ID", "DATE"]).copy()
    g = out.groupby("ID", sort=False)

    for col in sensor_cols:
        lag = g[col].shift(1)
        out[f"{col}_lag1"] = lag
        out[f"{col}_roll7_mean"] = lag.rolling(7, min_periods=2).mean().reset_index(level=0, drop=True)

    return out


def time_split_3way(
    df: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Require train_fraction > 0, val_fraction > 0, and train_fraction + val_fraction < 1")

    train_cutoff = df["DATE"].quantile(train_fraction)
    val_cutoff = df["DATE"].quantile(train_fraction + val_fraction)

    train_df = df[df["DATE"] <= train_cutoff].copy()
    val_df = df[(df["DATE"] > train_cutoff) & (df["DATE"] <= val_cutoff)].copy()
    test_df = df[df["DATE"] > val_cutoff].copy()
    return train_df, val_df, test_df


def evaluate_multiclass(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Result:
    # "Close failure" focuses on classes NOW, D1_7, D8_30 (0,1,2).
    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    close_recall_macro = float(np.mean(recalls))

    return Result(
        name=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        close_recall_macro=close_recall_macro,
    )


def print_distribution(label: str, y: np.ndarray) -> None:
    counts = pd.Series(y).value_counts().sort_index()
    total = len(y)
    print(f"{label} class distribution:")
    for cls, cnt in counts.items():
        name = CLASS_NAMES.get(int(cls), str(cls))
        print(f"  {int(cls)} ({name}): {int(cnt):,} ({cnt / total:.3%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-split failure-horizon multiclass prediction")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to CSV")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    args = parser.parse_args()

    raw = pd.read_csv(args.csv)

    required = {
        "ID",
        "DATE",
        "REGION_CLUSTER",
        "MAINTENANCE_VENDOR",
        "MANUFACTURER",
        "WELL_GROUP",
        "S5",
        "S8",
        "S13",
        "S15",
        "S16",
        "S17",
        "S18",
        "S19",
        "AGE_OF_EQUIPMENT",
        "EQUIPMENT_FAILURE",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    sensor_cols = ["S5", "S8", "S13", "S15", "S16", "S17", "S18", "S19"]

    data = build_rul_df(raw)
    data = add_horizon_target(data)
    data = add_leak_safe_features(data, sensor_cols=sensor_cols)

    train_df, val_df, test_df = time_split_3way(
        data,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    if args.max_train_rows > 0 and len(train_df) > args.max_train_rows:
        train_df = train_df.sample(n=args.max_train_rows, random_state=42).sort_values("DATE").reset_index(drop=True)

    feature_cols = [
        "ID",
        "REGION_CLUSTER",
        "MAINTENANCE_VENDOR",
        "MANUFACTURER",
        "WELL_GROUP",
        "AGE_OF_EQUIPMENT",
    ] + sensor_cols + [f"{s}_lag1" for s in sensor_cols] + [f"{s}_roll7_mean" for s in sensor_cols]

    categorical_cols = ["ID", "REGION_CLUSTER", "MAINTENANCE_VENDOR", "MANUFACTURER"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["FAILURE_HORIZON_CLASS"].to_numpy(dtype=int)
    X_val = val_df[feature_cols]
    y_val = val_df["FAILURE_HORIZON_CLASS"].to_numpy(dtype=int)
    X_test = test_df[feature_cols]
    y_test = test_df["FAILURE_HORIZON_CLASS"].to_numpy(dtype=int)

    # Very simple baseline: per-ID most common class from training data.
    id_mode = train_df.groupby("ID")["FAILURE_HORIZON_CLASS"].agg(lambda s: int(s.mode().iloc[0]))
    global_mode = int(train_df["FAILURE_HORIZON_CLASS"].mode().iloc[0])
    baseline_pred = X_test["ID"].map(id_mode).fillna(global_mode).to_numpy(dtype=int)
    baseline_result = evaluate_multiclass("Baseline (ID mode class)", y_test, baseline_pred)

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )

    lr_balanced = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        random_state=42,
    )

    lr_cost_sensitive = LogisticRegression(
        max_iter=2000,
        class_weight=CLASS_WEIGHT_NEAR_FAILURE,
        solver="saga",
        n_jobs=-1,
        random_state=42,
    )

    rf_cost_sensitive = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight=CLASS_WEIGHT_NEAR_FAILURE,
    )

    candidates = [
        ("LogReg balanced", lr_balanced),
        ("LogReg cost-sensitive", lr_cost_sensitive),
        ("RF cost-sensitive", rf_cost_sensitive),
    ]

    val_results: list[tuple[Result, Pipeline]] = []
    for name, model in candidates:
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_val)
        val_result = evaluate_multiclass(f"{name} (val)", y_val, y_val_pred)
        val_results.append((val_result, pipe))

    # Select model that is best at near-failure classes on validation.
    val_results.sort(key=lambda x: x[0].close_recall_macro, reverse=True)
    best_val_result, best_pipe = val_results[0]
    y_test_pred = best_pipe.predict(X_test)
    model_result = evaluate_multiclass(f"Selected model: {best_val_result.name.replace(' (val)', '')}", y_test, y_test_pred)

    print("=== Failure Horizon (Time-Split Multiclass) ===")
    print("Classes: 0=NOW, 1=D1_7, 2=D8_30, 3=GT_30")
    print(f"Rows: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    print_distribution("Train", y_train)
    print_distribution("Val", y_val)
    print_distribution("Test", y_test)

    print("\n1) Baseline")
    print(f"{baseline_result.name}")
    print(f"  accuracy:    {baseline_result.accuracy:.3f}")
    print(f"  macro_f1:    {baseline_result.macro_f1:.3f}")
    print(f"  weighted_f1: {baseline_result.weighted_f1:.3f}")
    print(f"  close_recall_macro (NOW,D1_7,D8_30): {baseline_result.close_recall_macro:.3f}")

    print("\n2) Validation model selection (higher close_recall_macro is better)")
    for r, _ in val_results:
        print(
            f"{r.name}: close_recall_macro={r.close_recall_macro:.3f}, "
            f"macro_f1={r.macro_f1:.3f}, accuracy={r.accuracy:.3f}"
        )

    print("\n3) Selected model on TEST")
    print(f"{model_result.name}")
    print(f"  accuracy:    {model_result.accuracy:.3f}")
    print(f"  macro_f1:    {model_result.macro_f1:.3f}")
    print(f"  weighted_f1: {model_result.weighted_f1:.3f}")
    print(f"  close_recall_macro (NOW,D1_7,D8_30): {model_result.close_recall_macro:.3f}")

    print("\nModel per-class report on test:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=[0, 1, 2, 3],
            target_names=[CLASS_NAMES[i] for i in [0, 1, 2, 3]],
            zero_division=0,
            digits=3,
        )
    )

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2, 3])
    print("Confusion matrix (rows=true, cols=pred, order: NOW,D1_7,D8_30,GT_30):")
    print(cm)


if __name__ == "__main__":
    main()
