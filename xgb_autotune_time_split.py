"""
Time-split XGBoost autotuning for binary near-failure tasks.

This script tunes XGBoost hyperparameters on validation data, then evaluates the
best configuration on the held-out test split.

Binary target definition per task:
- NOW: y = 1 if RUL_DAYS <= 0, else 0
- LE_7_DAYS: y = 1 if RUL_DAYS <= 7, else 0
- LE_30_DAYS: y = 1 if RUL_DAYS <= 30, else 0
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None


CSV_PATH = "equipment_failure_data_1.csv"
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_MAX_TRAIN_ROWS = 0
DEFAULT_RANDOM_STATE = 42
DEFAULT_PRINT_SEPARATOR = 72

# Features
DEFAULT_SENSOR_COLS = ("S5", "S8", "S13", "S15", "S16", "S17", "S18", "S19")
DEFAULT_ROLLING_WINDOW = 7
DEFAULT_ROLLING_MIN_PERIODS = 2
DEFAULT_BASE_FEATURE_COLS = (
    "REGION_CLUSTER",
    "MAINTENANCE_VENDOR",
    "MANUFACTURER",
    "WELL_GROUP",
    "AGE_OF_EQUIPMENT",
)
DEFAULT_CATEGORICAL_COLS = ("REGION_CLUSTER", "MAINTENANCE_VENDOR", "MANUFACTURER")

# Tasks: (task_name, max_rul_days_inclusive)
TASK_DEFINITIONS = {
    "NOW": 0,
    "LE_7_DAYS": 7,
    "LE_30_DAYS": 30,
}

# Autotune behavior
DEFAULT_TUNE_MODE = "single"  # "single" or "all"
DEFAULT_TUNE_MODE_CHOICES = ("single", "all")
DEFAULT_SELECTED_TASK = "NOW"
DEFAULT_N_TRIALS = 24
DEFAULT_SELECTION_METRIC = "pr_auc"  # "pr_auc" or "fbeta"
DEFAULT_SELECTION_METRIC_CHOICES = ("pr_auc", "fbeta")
DEFAULT_F_BETA = 2.0
DEFAULT_USE_AUTO_SCALE_POS_WEIGHT = True

# Search space (sampled randomly each trial)
GRID_N_ESTIMATORS = [250, 400, 600, 900]
GRID_MAX_DEPTH = [3, 4, 5, 6]
GRID_LEARNING_RATE = [0.02, 0.04, 0.06, 0.08]
GRID_MIN_CHILD_WEIGHT = [1, 3, 5, 8]
GRID_GAMMA = [0.0, 0.5, 1.0, 2.0]
GRID_SUBSAMPLE = [0.7, 0.85, 1.0]
GRID_COLSAMPLE_BYTREE = [0.6, 0.8, 1.0]
GRID_REG_ALPHA = [0.0, 0.1, 0.5]
GRID_REG_LAMBDA = [1.0, 3.0, 8.0]
GRID_SCALE_POS_WEIGHT_MULT = [0.75, 1.0, 1.25, 1.5]

# XGBoost fixed runtime settings.
DEFAULT_XGB_OBJECTIVE = "binary:logistic"
DEFAULT_XGB_EVAL_METRIC = "logloss"
DEFAULT_XGB_N_JOBS = -1

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
class TrialResult:
    score: float
    params: dict[str, float | int]
    threshold: float


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


def _sample_xgb_params(rng: random.Random, scale_pos_weight: float) -> dict[str, float | int]:
    return {
        "n_estimators": rng.choice(GRID_N_ESTIMATORS),
        "max_depth": rng.choice(GRID_MAX_DEPTH),
        "learning_rate": rng.choice(GRID_LEARNING_RATE),
        "min_child_weight": rng.choice(GRID_MIN_CHILD_WEIGHT),
        "gamma": rng.choice(GRID_GAMMA),
        "subsample": rng.choice(GRID_SUBSAMPLE),
        "colsample_bytree": rng.choice(GRID_COLSAMPLE_BYTREE),
        "reg_alpha": rng.choice(GRID_REG_ALPHA),
        "reg_lambda": rng.choice(GRID_REG_LAMBDA),
        "scale_pos_weight": scale_pos_weight,
    }


def _build_xgb(params: dict[str, float | int], random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective=DEFAULT_XGB_OBJECTIVE,
        eval_metric=DEFAULT_XGB_EVAL_METRIC,
        random_state=random_state,
        n_jobs=DEFAULT_XGB_N_JOBS,
        **params,
    )


def _evaluate_selection_metric(metric: str, beta: float, y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if metric == "pr_auc":
        threshold = tune_threshold(y_true, scores, beta=beta)
        return float(average_precision_score(y_true, scores)), threshold

    if metric == "fbeta":
        threshold = tune_threshold(y_true, scores, beta=beta)
        y_pred = (scores >= threshold).astype(int)
        return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)), threshold

    raise ValueError(f"Unsupported selection metric: {metric}")


def tune_for_task(
    task_name: str,
    max_days: int,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    pre: ColumnTransformer,
    n_trials: int,
    metric: str,
    beta: float,
    random_state: int,
    use_auto_spw: bool,
) -> tuple[TrialResult, np.ndarray, np.ndarray]:
    rng = random.Random(random_state + max_days)

    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    base_spw = (neg / max(pos, 1)) if use_auto_spw else 1.0

    best: TrialResult | None = None

    for i in range(1, n_trials + 1):
        mult = rng.choice(GRID_SCALE_POS_WEIGHT_MULT) if use_auto_spw else 1.0
        params = _sample_xgb_params(rng, scale_pos_weight=float(base_spw * mult))

        clf = _build_xgb(params, random_state=random_state)
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)

        val_scores = pipe.predict_proba(X_val)[:, 1]
        score, threshold = _evaluate_selection_metric(metric, beta, y_val, val_scores)

        if best is None or score > best.score:
            best = TrialResult(score=score, params=params, threshold=threshold)

        print(f"  trial {i:02d}/{n_trials} score={score:.4f} thr={threshold:.4f}")

    if best is None:
        raise RuntimeError("No successful trial was completed.")

    best_model = _build_xgb(best.params, random_state=random_state)
    best_pipe = Pipeline(steps=[("pre", pre), ("clf", best_model)])
    best_pipe.fit(X_train, y_train)

    val_scores_best = best_pipe.predict_proba(X_val)[:, 1]
    test_scores_best = best_pipe.predict_proba(X_test)[:, 1]

    print(f"\nBest params for {task_name}: {best.params}")
    print(f"Best validation {metric}: {best.score:.4f}")
    print(f"Best threshold (val, F-beta): {best.threshold:.4f}")

    return best, val_scores_best, test_scores_best


def evaluate_test(y_true: np.ndarray, y_scores: np.ndarray, threshold: float, beta: float) -> dict[str, float]:
    y_pred = (y_scores >= threshold).astype(int)
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fbeta": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "alert_rate": float(np.mean(y_pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def main() -> None:
    if XGBClassifier is None:
        raise ModuleNotFoundError("xgboost is required for this script. Install with `pip install xgboost`.")

    parser = argparse.ArgumentParser(description="Autotune XGBoost for time-split binary failure tasks")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument("--tune-mode", choices=DEFAULT_TUNE_MODE_CHOICES, default=DEFAULT_TUNE_MODE)
    parser.add_argument("--task", choices=list(TASK_DEFINITIONS.keys()), default=DEFAULT_SELECTED_TASK)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--metric", choices=DEFAULT_SELECTION_METRIC_CHOICES, default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--beta", type=float, default=DEFAULT_F_BETA)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--no-auto-scale-pos-weight", action="store_true")
    args = parser.parse_args()

    raw = pd.read_csv(args.csv)
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    sensor_cols = list(DEFAULT_SENSOR_COLS)
    data = add_leak_safe_features(build_rul_df(raw), sensor_cols=sensor_cols)
    train_df, val_df, test_df = time_split_3way(data, train_fraction=args.train_fraction, val_fraction=args.val_fraction)

    if args.max_train_rows > 0 and len(train_df) > args.max_train_rows:
        train_df = train_df.sample(n=args.max_train_rows, random_state=args.random_state).sort_values("DATE").reset_index(drop=True)

    feature_cols = list(DEFAULT_BASE_FEATURE_COLS) + sensor_cols + [f"{s}_lag1" for s in sensor_cols] + [
        f"{s}_roll7_mean" for s in sensor_cols
    ]

    categorical_cols = list(DEFAULT_CATEGORICAL_COLS)
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

    tasks = [args.task] if args.tune_mode == "single" else list(TASK_DEFINITIONS.keys())

    print("=== XGBoost Autotune (Time Split) ===")
    print(f"Rows: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    print(f"Tune mode: {args.tune_mode}")
    print(f"Trials per task: {args.n_trials}")
    print(f"Selection metric: {args.metric}")
    print(f"Threshold metric: F-beta (beta={args.beta})")

    for task_name in tasks:
        max_days = TASK_DEFINITIONS[task_name]
        y_train = (train_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        y_val = (val_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        y_test = (test_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)

        print("\n" + "-" * DEFAULT_PRINT_SEPARATOR)
        print(f"Task: {task_name}")
        print(f"Binary target used: y = 1 if RUL_DAYS <= {max_days}, else 0")
        print(f"Positive rate train/val/test: {y_train.mean():.4f} / {y_val.mean():.4f} / {y_test.mean():.4f}")

        best, _, test_scores = tune_for_task(
            task_name=task_name,
            max_days=max_days,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            pre=pre,
            n_trials=args.n_trials,
            metric=args.metric,
            beta=args.beta,
            random_state=args.random_state,
            use_auto_spw=not args.no_auto_scale_pos_weight,
        )

        test_metrics = evaluate_test(y_test, test_scores, threshold=best.threshold, beta=args.beta)
        print("Test metrics with tuned params and tuned threshold:")
        print(
            "  precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
            "fbeta={fbeta:.4f} pr_auc={pr_auc:.4f} roc_auc={roc_auc:.4f} alert_rate={alert_rate:.4f}".format(**test_metrics)
        )


if __name__ == "__main__":
    main()
