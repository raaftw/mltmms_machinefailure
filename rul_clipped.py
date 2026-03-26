"""
Predict imminent failure (binary) from daily machine data.

Target:
- IMMINENT_FAILURE = 1 if failure occurs within `--imminent-days` (inclusive).
- IMMINENT_FAILURE = 0 otherwise.

This script compares:
1) Simple baseline: per-ID historical imminent rate.
2) Improved model: RandomForestClassifier with leak-safe lag/rolling features.

Evaluation is focused on imminent-failure detection quality (recall/precision/F1/F2, PR-AUC).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CSV_PATH = "equipment_failure_data_1.csv"

# Tuning/config defaults: change these in one place for quick experiments.
DEFAULT_IMMINENT_DAYS = 7  # Larger value labels more rows as positive; usually recall rises and precision drops.
DEFAULT_TRAIN_FRACTION = 0.70  # More train data can improve stability, but leaves less data for threshold tuning/test reliability.
DEFAULT_VAL_FRACTION = 0.15  # More validation data gives more reliable tuned threshold, but reduces training size.
DEFAULT_N_ESTIMATORS = 500  # More trees usually improve robustness/metrics slightly, but increase runtime.
DEFAULT_THRESHOLD = 0.1  # Lower threshold -> more alerts and higher recall, usually lower precision; higher does the opposite.
DEFAULT_THRESHOLD_BETA = 2.5  # Conservative policy: beta<1 prioritizes precision during threshold tuning.
DEFAULT_RF_RANDOM_STATE = 42  # Reproducibility only; changes randomness, not expected direction of metrics.
DEFAULT_RF_CLASS_WEIGHT = None  # Conservative policy: avoid extra minority up-weighting to reduce false alarms.
DEFAULT_SENSOR_COLS = ("S5", "S8", "S13", "S15", "S16", "S17", "S18", "S19")  # Adding useful sensors can improve PR-AUC; noisy sensors can hurt.
DEFAULT_ROLLING_WINDOW = 7  # Larger window smooths noise (often better precision), smaller window reacts faster (often better recall).
DEFAULT_ROLLING_MIN_PERIODS = 2  # Lower value creates earlier features with less history (noisier); higher is stabler but drops early-signal info.


def info_print(message: str) -> None:
    # flush=True ensures progress is visible immediately in the terminal.
    print(f"[INFO] {message}", flush=True)


def debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[DEBUG] {message}", flush=True)


@dataclass
class ModelResult:
    name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    f2: float
    pr_auc: float
    roc_auc: float
    predicted_positive_rate: float


def build_rul_clipped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-row target:
      RUL_DAYS = first_failure_date_for_id - DATE
      RUL_100 = clip(RUL_DAYS, upper=100)

    Keep only rows on/before first failure to avoid post-failure leakage.
    """
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
    out["RUL_100"] = out["RUL_DAYS"].clip(lower=0, upper=100).astype(int)

    return out.sort_values(["ID", "DATE"]).reset_index(drop=True)


def add_imminent_target(df: pd.DataFrame, imminent_days: int) -> pd.DataFrame:
    out = df.copy()
    out["IMMINENT_FAILURE"] = (out["RUL_DAYS"] <= imminent_days).astype(int)
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


def add_basic_time_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    rolling_min_periods: int = DEFAULT_ROLLING_MIN_PERIODS,
) -> pd.DataFrame:
    """
    Add simple leak-safe features:
    - sensor lag1
    - sensor 7-day rolling mean from past values only
    """
    out = df.sort_values(["ID", "DATE"]).copy()
    grouped = out.groupby("ID", sort=False)

    for col in sensor_cols:
        lag = grouped[col].shift(1)
        out[f"{col}_lag1"] = lag
        out[f"{col}_roll7_mean"] = (
            lag.rolling(window=rolling_window, min_periods=rolling_min_periods).mean().reset_index(level=0, drop=True)
        )

    return out


def pick_threshold_by_fbeta(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5

    fbeta = (1 + beta**2) * precision[:-1] * recall[:-1] / ((beta**2 * precision[:-1]) + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(fbeta))
    return float(thresholds[best_idx])


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float, name: str) -> ModelResult:
    y_pred = (y_score >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_score))
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = float("nan")

    return ModelResult(
        name=name,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        f2=f2,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        predicted_positive_rate=float(y_pred.mean()),
    )


def run_simple_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[ModelResult, ModelResult, np.ndarray, np.ndarray]:
    # Baseline risk score is the historical imminent rate by ID from train split only.
    id_rate = train_df.groupby("ID")["IMMINENT_FAILURE"].mean()
    global_rate = float(train_df["IMMINENT_FAILURE"].mean())

    y_val = val_df["IMMINENT_FAILURE"].to_numpy(dtype=int)
    val_score = val_df["ID"].map(id_rate).fillna(global_rate).to_numpy(dtype=float)
    tuned_threshold = pick_threshold_by_fbeta(y_val, val_score, beta=DEFAULT_THRESHOLD_BETA)

    y_test = test_df["IMMINENT_FAILURE"].to_numpy(dtype=int)
    test_score = test_df["ID"].map(id_rate).fillna(global_rate).to_numpy(dtype=float)

    at_default = evaluate_binary(
        y_test,
        test_score,
        threshold=DEFAULT_THRESHOLD,
        name=f"Simple baseline (ID imminent rate) @{DEFAULT_THRESHOLD:.2f}",
    )
    at_tuned = evaluate_binary(
        y_test,
        test_score,
        threshold=tuned_threshold,
        name="Simple baseline (ID imminent rate) @tuned",
    )
    return at_default, at_tuned, y_test, test_score


def run_improved_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_estimators: int,
) -> tuple[ModelResult, ModelResult, np.ndarray, np.ndarray]:
    sensor_cols = list(DEFAULT_SENSOR_COLS)

    train_feat = add_basic_time_features(train_df, sensor_cols=sensor_cols)
    val_feat = add_basic_time_features(val_df, sensor_cols=sensor_cols)
    test_feat = add_basic_time_features(test_df, sensor_cols=sensor_cols)

    base_features = [
        "ID",
        "REGION_CLUSTER",
        "MAINTENANCE_VENDOR",
        "MANUFACTURER",
        "WELL_GROUP",
        "AGE_OF_EQUIPMENT",
    ] + sensor_cols

    extra_features: list[str] = []
    for col in sensor_cols:
        extra_features.append(f"{col}_lag1")
        extra_features.append(f"{col}_roll7_mean")

    features = base_features + extra_features

    X_train = train_feat[features]
    y_train = train_feat["IMMINENT_FAILURE"].to_numpy(dtype=int)
    X_val = val_feat[features]
    y_val = val_feat["IMMINENT_FAILURE"].to_numpy(dtype=int)
    X_test = test_feat[features]
    y_test = test_feat["IMMINENT_FAILURE"].to_numpy(dtype=int)

    categorical_cols = ["ID", "REGION_CLUSTER", "MAINTENANCE_VENDOR", "MANUFACTURER"]
    numeric_cols = [c for c in features if c not in categorical_cols]

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
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=DEFAULT_RF_RANDOM_STATE,
        n_jobs=-1,
        class_weight=DEFAULT_RF_CLASS_WEIGHT,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    val_score = pipe.predict_proba(X_val)[:, 1]
    test_score = pipe.predict_proba(X_test)[:, 1]
    tuned_threshold = pick_threshold_by_fbeta(y_val, val_score, beta=DEFAULT_THRESHOLD_BETA)

    at_default = evaluate_binary(
        y_test,
        test_score,
        threshold=DEFAULT_THRESHOLD,
        name=f"Improved RF @{DEFAULT_THRESHOLD:.2f}",
    )
    at_tuned = evaluate_binary(y_test, test_score, threshold=tuned_threshold, name="Improved RF @tuned")
    return at_default, at_tuned, y_test, test_score


def print_result(result: ModelResult) -> None:
    print(f"{result.name}")
    print(f"  threshold: {result.threshold:.4f}")
    print(f"  precision: {result.precision:.3f}")
    print(f"  recall:    {result.recall:.3f}")
    print(f"  f1:        {result.f1:.3f}")
    print(f"  f2:        {result.f2:.3f}")
    print(f"  pr_auc:    {result.pr_auc:.3f}")
    print(f"  roc_auc:   {result.roc_auc:.3f}")
    print(f"  alert_rate:{result.predicted_positive_rate:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Imminent failure prediction")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to CSV")
    parser.add_argument(
        "--imminent-days",
        type=int,
        default=DEFAULT_IMMINENT_DAYS,
        help="Positive class if failure occurs within this many days",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Fraction of earliest dates for train",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of middle dates for validation",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Number of trees for RandomForestClassifier",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    args = parser.parse_args()

    if args.n_estimators < 1 or args.imminent_days < 1:
        raise ValueError("--n-estimators and --imminent-days must be >= 1")

    debug = args.debug
    run_start = time.perf_counter()

    info_print("Starting imminent-failure pipeline")
    debug_print(debug, f"Loading CSV from: {args.csv}")
    t0 = time.perf_counter()
    raw = pd.read_csv(args.csv)
    info_print(f"CSV loaded in {time.perf_counter() - t0:.2f}s")
    debug_print(debug, f"Raw shape: {raw.shape}")

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
    debug_print(debug, "All required columns found")

    info_print("Building clipped RUL target")
    debug_print(debug, "Building clipped RUL target and imminent label")
    t0 = time.perf_counter()
    data = add_imminent_target(build_rul_clipped(raw), imminent_days=args.imminent_days)
    info_print(f"Target built in {time.perf_counter() - t0:.2f}s")
    debug_print(
        debug,
        (
            f"Prepared rows: {len(data):,}, IDs: {data['ID'].nunique():,}, "
            f"RUL_100 range: {data['RUL_100'].min()}..{data['RUL_100'].max()}, "
            f"imminent positive rate: {data['IMMINENT_FAILURE'].mean():.4f}"
        ),
    )

    info_print("Running time split")
    debug_print(
        debug,
        f"Running time split with train_fraction={args.train_fraction}, val_fraction={args.val_fraction}",
    )
    t0 = time.perf_counter()
    train_df, val_df, test_df = time_split_3way(
        data,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    info_print(f"Split completed in {time.perf_counter() - t0:.2f}s")
    debug_print(debug, f"Train rows: {len(train_df):,}, Val rows: {len(val_df):,}, Test rows: {len(test_df):,}")
    debug_print(debug, f"Train positive rate: {train_df['IMMINENT_FAILURE'].mean():.4f}")
    debug_print(debug, f"Val positive rate: {val_df['IMMINENT_FAILURE'].mean():.4f}")
    debug_print(debug, f"Test positive rate: {test_df['IMMINENT_FAILURE'].mean():.4f}")

    info_print("Training simple baseline")
    debug_print(debug, "Training simple baseline")
    t0 = time.perf_counter()
    simple_default, simple_tuned, y_test_simple, score_test_simple = run_simple_baseline(train_df, val_df, test_df)
    info_print(f"Simple baseline done in {time.perf_counter() - t0:.2f}s")

    info_print("Training improved model")
    debug_print(debug, "Training improved model")
    debug_print(debug, f"Improved model n_estimators={args.n_estimators}")
    t0 = time.perf_counter()
    improved_default, improved_tuned, y_test_rf, score_test_rf = run_improved_model(
        train_df,
        val_df,
        test_df,
        n_estimators=args.n_estimators,
    )
    info_print(f"Improved model done in {time.perf_counter() - t0:.2f}s")
    debug_print(debug, "Evaluation complete")
    info_print(f"Total runtime: {time.perf_counter() - run_start:.2f}s")

    print("=== Imminent Failure Prediction ===")
    print(f"Positive class: failure within <= {args.imminent_days} days")
    print(
        f"Rows total: {len(data):,} | Train: {len(train_df):,} | "
        f"Val: {len(val_df):,} | Test: {len(test_df):,}"
    )
    print(
        f"Positive rate | train={train_df['IMMINENT_FAILURE'].mean():.4f}, "
        f"val={val_df['IMMINENT_FAILURE'].mean():.4f}, "
        f"test={test_df['IMMINENT_FAILURE'].mean():.4f}"
    )

    print("\n1) Simple baseline")
    print_result(simple_default)
    print_result(simple_tuned)

    print("\n2) Improved model")
    print(f"n_estimators: {args.n_estimators}")
    print_result(improved_default)
    print_result(improved_tuned)

    if improved_tuned.precision > simple_tuned.precision:
        delta = improved_tuned.precision - simple_tuned.precision
        print(f"\nImprovement on tuned precision: +{delta:.3f}")
    else:
        print("\nNo tuned precision improvement over simple baseline on this split.")

    cm = confusion_matrix(y_test_rf, (score_test_rf >= improved_tuned.threshold).astype(int), labels=[0, 1])
    print("\nImproved tuned confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm)


if __name__ == "__main__":
    main()
