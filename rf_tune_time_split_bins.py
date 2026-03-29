"""
Time-split Random Forest tuning for one-vs-rest RUL bins.

Goal:
- Keep the same preprocessing logic as time_split_binary_horizons.py
- Let you tune Random Forest settings per bin
- Let you change bin cutoffs at the top of this file

Each bin is trained as a binary target:
- y = 1 if RUL_DAYS <= bin_max_days
- y = 0 otherwise
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# User-tunable top-level config
# -----------------------------
CSV_PATH = "equipment_failure_data_1.csv"
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_MAX_TRAIN_ROWS = 0

# Bin cutoffs for one-vs-rest tasks. Edit these to change bin size/coverage.
# Example: (0, 3, 7, 14, 30)
BIN_MAX_DAYS = (0, 7, 30)

# Threshold tuning metric.
# If USE_TASK_SPECIFIC_BETA=False, GLOBAL_BETA is used for all bins.
GLOBAL_BETA = 2.0
USE_TASK_SPECIFIC_BETA = True
TASK_BETA = {
    "LE_0_DAYS": 3.0,
    "LE_7_DAYS": 1.5,
    "LE_30_DAYS": 0.8,
}

# Selection metric for choosing best RF config per bin: "pr_auc" or "fbeta".
SELECTION_METRIC = "fbeta"
SELECTION_METRIC_CHOICES = ("pr_auc", "fbeta")

# Same feature settings as time_split_binary_horizons.py
DEFAULT_SENSOR_COLS = ("S5", "S8", "S13", "S15", "S16", "S17", "S18", "S19")
DEFAULT_ROLLING_WINDOW = 7
DEFAULT_ROLLING_MIN_PERIODS = 2

# Optional PCA stage (enabled to match current horizons script behavior).
USE_PCA = True
PCA_N_COMPONENTS = 0.95

# RF candidate configurations to try for each bin.
# Add/remove dictionaries to tune aggressively.
RF_PARAM_GRID = [
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    },
    {
        "n_estimators": 700,
        "max_depth": 14,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    },
    {
        "n_estimators": 900,
        "max_depth": 18,
        "min_samples_split": 6,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    },
    {
        "n_estimators": 1000,
        "max_depth": 22,
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "max_features": 0.7,
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    },
]

REQUIRED_COLUMNS = {
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


@dataclass
class TuneResult:
    task: str
    config_id: int
    threshold: float
    val_metric: float
    precision: float
    recall: float
    f1: float
    fbeta: float
    pr_auc: float
    roc_auc: float
    alert_rate: float
    params: dict


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


def add_leak_safe_features(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    out = df.sort_values(["ID", "DATE"]).copy()
    g = out.groupby("ID", sort=False)

    for col in sensor_cols:
        lag = g[col].shift(1)
        out[f"{col}_lag1"] = lag
        out[f"{col}_roll7_mean"] = (
            lag.rolling(DEFAULT_ROLLING_WINDOW, min_periods=DEFAULT_ROLLING_MIN_PERIODS)
            .mean()
            .reset_index(level=0, drop=True)
        )

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


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, beta: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5

    fbeta = (1 + beta**2) * precision[:-1] * recall[:-1] / ((beta**2 * precision[:-1]) + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(fbeta))])


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float, beta: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fbeta": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "alert_rate": float(y_pred.mean()),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def build_rf_pipeline(pre: ColumnTransformer, rf_params: dict) -> Pipeline:
    steps: list[tuple[str, object]] = [("pre", pre)]
    if USE_PCA:
        steps.append(("pca", PCA(n_components=PCA_N_COMPONENTS, random_state=42)))
    steps.append(("clf", RandomForestClassifier(**rf_params)))
    return Pipeline(steps=steps)


def _make_task_definitions(bin_max_days: tuple[int, ...]) -> list[tuple[str, int]]:
    clean_bins = sorted(set(int(x) for x in bin_max_days))
    return [(f"LE_{days}_DAYS", days) for days in clean_bins]


def _selection_score(metric: str, y_true: np.ndarray, y_scores: np.ndarray, threshold: float, beta: float) -> float:
    if metric == "pr_auc":
        return float(average_precision_score(y_true, y_scores))
    if metric == "fbeta":
        y_pred = (y_scores >= threshold).astype(int)
        return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
    raise ValueError(f"Unsupported selection metric: {metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Random Forest per RUL bin with time split")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument("--metric", choices=SELECTION_METRIC_CHOICES, default=SELECTION_METRIC)
    parser.add_argument(
        "--bin-max-days",
        type=int,
        nargs="+",
        default=list(BIN_MAX_DAYS),
        help="Override bin cutoffs, e.g. --bin-max-days 0 3 7 14 30",
    )
    parser.add_argument("--beta", type=float, default=GLOBAL_BETA)
    args = parser.parse_args()

    raw = pd.read_csv(args.csv)
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    sensor_cols = list(DEFAULT_SENSOR_COLS)
    data = add_leak_safe_features(build_rul_df(raw), sensor_cols=sensor_cols)
    train_df, val_df, test_df = time_split_3way(
        data,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    if args.max_train_rows > 0 and len(train_df) > args.max_train_rows:
        train_df = train_df.sample(n=args.max_train_rows, random_state=42).sort_values("DATE").reset_index(drop=True)

    feature_cols = [
        "REGION_CLUSTER",
        "MAINTENANCE_VENDOR",
        "MANUFACTURER",
        "WELL_GROUP",
        "AGE_OF_EQUIPMENT",
    ] + sensor_cols + [f"{s}_lag1" for s in sensor_cols] + [f"{s}_roll7_mean" for s in sensor_cols]

    categorical_cols = ["REGION_CLUSTER", "MAINTENANCE_VENDOR", "MANUFACTURER"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

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

    task_definitions = _make_task_definitions(tuple(args.bin_max_days))
    if len(task_definitions) == 0:
        raise ValueError("At least one bin must be provided in BIN_MAX_DAYS or --bin-max-days")

    print("=== Time-Split RF Bin Tuning ===")
    print(f"Rows: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    print(f"Metric used to pick best config: {args.metric}")
    print(f"Bins: {[d for _, d in task_definitions]}")
    print(f"RF candidate configs: {len(RF_PARAM_GRID)}")
    print(f"PCA enabled: {USE_PCA} (n_components={PCA_N_COMPONENTS})")

    all_rows: list[TuneResult] = []

    for task_name, max_days in task_definitions:
        y_train = (train_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        y_val = (val_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        y_test = (test_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)

        beta_task = TASK_BETA.get(task_name, args.beta) if USE_TASK_SPECIFIC_BETA else args.beta

        print(f"\nTask {task_name}: RUL <= {max_days}")
        print(f"Positive rates: train={float(np.mean(y_train)):.4f}, val={float(np.mean(y_val)):.4f}, test={float(np.mean(y_test)):.4f}")

        best_result: TuneResult | None = None

        for idx, params in enumerate(RF_PARAM_GRID, start=1):
            pipe = build_rf_pipeline(pre, params)
            pipe.fit(X_train, y_train)

            val_scores = pipe.predict_proba(X_val)[:, 1]
            test_scores = pipe.predict_proba(X_test)[:, 1]
            threshold = tune_threshold(y_val, val_scores, beta=beta_task)

            val_metric = _selection_score(args.metric, y_val, val_scores, threshold=threshold, beta=beta_task)
            test_metrics = evaluate_binary(y_test, test_scores, threshold=threshold, beta=beta_task)

            row = TuneResult(
                task=task_name,
                config_id=idx,
                threshold=threshold,
                val_metric=val_metric,
                precision=test_metrics["precision"],
                recall=test_metrics["recall"],
                f1=test_metrics["f1"],
                fbeta=test_metrics["fbeta"],
                pr_auc=test_metrics["pr_auc"],
                roc_auc=test_metrics["roc_auc"],
                alert_rate=test_metrics["alert_rate"],
                params=params,
            )
            all_rows.append(row)

            print(f"  config {idx:02d}: val_{args.metric}={val_metric:.4f}, thr={threshold:.4f}, test_fbeta={row.fbeta:.4f}")

            if best_result is None or row.val_metric > best_result.val_metric:
                best_result = row

        if best_result is None:
            raise RuntimeError(f"No successful RF config for task {task_name}")

        print("  Best config for this task:")
        print(f"    config_id={best_result.config_id} val_{args.metric}={best_result.val_metric:.4f}")
        print(f"    threshold={best_result.threshold:.4f}")
        print(f"    test metrics: precision={best_result.precision:.4f} recall={best_result.recall:.4f} "
              f"f1={best_result.f1:.4f} fbeta={best_result.fbeta:.4f} pr_auc={best_result.pr_auc:.4f}")
        print(f"    params={best_result.params}")

    out = pd.DataFrame([r.__dict__ for r in all_rows])
    summary = out.sort_values(["task", "val_metric"], ascending=[True, False]).groupby("task", as_index=False).head(1)

    print("\n=== Best Per Bin (by validation metric) ===")
    print(
        summary[
            [
                "task",
                "config_id",
                "val_metric",
                "threshold",
                "precision",
                "recall",
                "f1",
                "fbeta",
                "pr_auc",
                "roc_auc",
                "alert_rate",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
