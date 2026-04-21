"""
Time-split XGBoost training for three cumulative failure horizons.

Trains three separate binary XGBoost models with separate hyperparameters:
- LE_0_DAYS:  y = 1 if RUL_DAYS <= 0
- LE_7_DAYS:  y = 1 if RUL_DAYS <= 7
- LE_30_DAYS: y = 1 if RUL_DAYS <= 30

Also builds a combined 4-class horizon output from the three cumulative probabilities:
- 0: NOW    (RUL == 0)
- 1: D1_7   (1 <= RUL <= 7)
- 2: D8_30  (8 <= RUL <= 30)
- 3: GT_30  (RUL > 30)
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
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


# -----------------------------
# User-tunable top-level config
# -----------------------------
CSV_PATH = "equipment_failure_data_1.csv"
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_MAX_TRAIN_ROWS = 0
RANDOM_STATE = 42
DEFAULT_SAVE_COMBINED_PLOTS = True
DEFAULT_PLOT_OUTPUT_DIR = "plots"
DEFAULT_SHOW_COMBINED_PLOTS = False

# Horizons requested by user.
BIN_MAX_DAYS = (0, 7, 30)
DEFAULT_TASK_DEFINITIONS = (
    ("NOW", 0),
    ("LE_7_DAYS", 7),
    ("LE_30_DAYS", 30),
)

# Threshold tuning metric.
GLOBAL_BETA = 2.0
USE_TASK_SPECIFIC_BETA = True
TASK_BETA = {
    "NOW": 3.0,
    "LE_7_DAYS": 1.5,
    "LE_30_DAYS": 0.8,
}

# Same feature settings as other time-split scripts.
DEFAULT_SENSOR_COLS = ("S5", "S8", "S13", "S15", "S16", "S17", "S18", "S19")
DEFAULT_BASE_FEATURE_COLS = (
    "REGION_CLUSTER",
    "MAINTENANCE_VENDOR",
    "MANUFACTURER",
    "WELL_GROUP",
    "AGE_OF_EQUIPMENT",
)
DEFAULT_CATEGORICAL_COLS = ("REGION_CLUSTER", "MAINTENANCE_VENDOR", "MANUFACTURER")
DEFAULT_ROLLING_WINDOW = 7
DEFAULT_ROLLING_MIN_PERIODS = 2

# Optional PCA stage.
USE_PCA = True
PCA_N_COMPONENTS = 0.95

# Preprocessing settings.
CATEGORICAL_IMPUTER_STRATEGY = "most_frequent"
NUMERIC_IMPUTER_STRATEGY = "median"
ONEHOT_HANDLE_UNKNOWN = "ignore"

# XGBoost shared runtime settings.
DEFAULT_XGB_SUBSAMPLE = 0.85
DEFAULT_XGB_COLSAMPLE_BYTREE = 0.85
DEFAULT_XGB_REG_LAMBDA = 1.0
DEFAULT_XGB_RANDOM_STATE = 42
DEFAULT_XGB_N_JOBS = -1
DEFAULT_XGB_EVAL_METRIC = "logloss"

# Combined horizon class settings.
HORIZON_CLASS_LABELS = [0, 1, 2, 3]
HORIZON_CLASS_NAMES = ["NOW", "D1_7", "D8_30", "GT_30"]

# Deployment policy thresholds for priority rule LE_0_DAYS -> LE_7_DAYS -> LE_30_DAYS -> GT_30.
# These are the tuned deployment thresholds suggested by the current validation results.
DEFAULT_DEPLOY_THRESHOLDS = {
    "NOW": 0.0010621459223330021,
    "LE_7_DAYS": 0.011700638569891453,
    "LE_30_DAYS": 0.0003786543384194374,
}

# Automatic combination tuning (validation-based).
AUTO_TUNE_COMBINATION = False
COMBINATION_SCORE_METRIC = "priority_weighted_f1"  # prioritize F1 in order: NOW > D1_7 > D8_30 > GT_30
COMBINATION_SCORE_CHOICES = ("priority_weighted_f1", "macro_f1", "weighted_f1", "macro_recall", "accuracy")
COMBINATION_CLASS_WEIGHTS = {
    0: 0.45,  # NOW - highest priority
    1: 0.30,  # D1_7 - second priority
    2: 0.20,  # D8_30 - third priority
    3: 0.05,  # GT_30 - outside / lowest priority
}
COMBINATION_THRESHOLD_GRID = {
    "NOW": tuple(np.logspace(-4, -1.5, 20)),  # Fine-grained logarithmic search
    "LE_7_DAYS": tuple(np.logspace(-4, -1.5, 25)),
    "LE_30_DAYS": tuple(np.logspace(-6, -2, 20)),
}

# Constrained combination tuning to avoid collapse of mid-range classes.
ENFORCE_COMBINATION_RECALL_FLOORS = True
COMBINATION_MIN_CLASS_RECALL = {
    0: 0.28,  # NOW - maintain reasonable early warning
    1: 0.22,  # D1_7 - strong mid-range target
    2: 0.08,  # D8_30 - balanced with other early classes
    3: 0.0,   # GT_30 - no floor
}

# Separate XGBoost settings per horizon model (user-provided best params).
XGB_PARAMS_BY_TASK = {
    "NOW": {
        "n_estimators": 261,
        "max_depth": 15,
        "learning_rate": 0.005,
    },
    "LE_7_DAYS": {
        "n_estimators": 261,
        "max_depth": 10,
        "learning_rate": 0.002,
    },
    "LE_30_DAYS": {
        "n_estimators": 703,
        "max_depth": 15,
        "learning_rate": 0.06,
    },
}

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
class BinaryResult:
    task: str
    threshold: float
    precision: float
    recall: float
    f1: float
    fbeta: float
    pr_auc: float
    roc_auc: float
    alert_rate: float
    positives_train: int
    positives_val: int
    positives_test: int


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


def build_true_horizon_class(rul_days: np.ndarray) -> np.ndarray:
    """Map RUL days to 4-class horizon label.

    0: NOW
    1: D1_7
    2: D8_30
    3: GT_30
    """
    out = np.full_like(rul_days, fill_value=3, dtype=int)
    out[rul_days == 0] = 0
    out[(rul_days >= 1) & (rul_days <= 7)] = 1
    out[(rul_days >= 8) & (rul_days <= 30)] = 2
    return out


def build_horizon_confidence_table(model_scores: dict[str, np.ndarray]) -> np.ndarray:
    """Convert cumulative binary probabilities into 4-class horizon confidences."""
    p_now_raw = model_scores["NOW"]
    p_le7_raw = model_scores["LE_7_DAYS"]
    p_le30_raw = model_scores["LE_30_DAYS"]

    # Enforce monotonic cumulative probabilities: P(<=0) <= P(<=7) <= P(<=30).
    p_now = np.minimum.reduce([p_now_raw, p_le7_raw, p_le30_raw])
    p_le7 = np.maximum(p_le7_raw, p_now)
    p_le7 = np.minimum(p_le7, p_le30_raw)
    p_le30 = np.maximum(p_le30_raw, p_le7)

    c_now = p_now
    c_d1_7 = np.maximum(p_le7 - p_now, 0.0)
    c_d8_30 = np.maximum(p_le30 - p_le7, 0.0)
    c_gt30 = np.maximum(1.0 - p_le30, 0.0)

    conf = np.column_stack([c_now, c_d1_7, c_d8_30, c_gt30])
    row_sum = conf.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return conf / row_sum


def predict_horizon_by_policy(model_scores: dict[str, np.ndarray], thresholds: dict[str, float]) -> np.ndarray:
    """Priority rule prediction with explicit deploy thresholds.

    0 (NOW)   if p_le0 >= t0
    1 (D1_7)  if p_le7 >= t7
    2 (D8_30) if p_le30 >= t30
    3 (GT_30) otherwise
    """
    p_le0 = model_scores["NOW"]
    p_le7 = model_scores["LE_7_DAYS"]
    p_le30 = model_scores["LE_30_DAYS"]

    pred = np.full(shape=p_le0.shape[0], fill_value=3, dtype=int)
    pred[p_le30 >= thresholds["LE_30_DAYS"]] = 2
    pred[p_le7 >= thresholds["LE_7_DAYS"]] = 1
    pred[p_le0 >= thresholds["NOW"]] = 0
    return pred


def predict_horizon_by_cascade(model_scores: dict[str, np.ndarray], thresholds: dict[str, float]) -> np.ndarray:
    """Sequential cascade prediction using tuned binary thresholds.

    Decision order:
    1) NOW
    2) D1_7
    3) D8_30
    4) GT_30

    Each later model only applies to samples rejected by earlier steps.
    """
    p_now = model_scores["NOW"]
    p_le7 = model_scores["LE_7_DAYS"]
    p_le30 = model_scores["LE_30_DAYS"]

    pred = np.full(shape=p_now.shape[0], fill_value=3, dtype=int)

    mask_now = p_now >= thresholds["NOW"]
    pred[mask_now] = 0

    remaining = ~mask_now
    mask_le7 = remaining & (p_le7 >= thresholds["LE_7_DAYS"])
    pred[mask_le7] = 1

    remaining = remaining & ~mask_le7
    mask_le30 = remaining & (p_le30 >= thresholds["LE_30_DAYS"])
    pred[mask_le30] = 2

    return pred


def print_cascade_stage_metrics(
    y_true_horizon: np.ndarray,
    model_scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> None:
    """Print conditional precision/recall for each cascade stage.

    Stage 2 metrics are computed only on samples that did not trigger the NOW gate.
    Stage 3 metrics are computed only on samples that did not trigger NOW or D1_7.
    """
    p_now = model_scores["NOW"]
    p_le7 = model_scores["LE_7_DAYS"]
    p_le30 = model_scores["LE_30_DAYS"]

    print("\n=== Cascade Stage Metrics (conditional on previous gates) ===")

    # Stage 1: NOW vs not-NOW on the full set.
    pred_now = p_now >= thresholds["NOW"]
    true_now = y_true_horizon == 0
    print(
        "Stage 1 NOW gate: "
        f"precision={precision_score(true_now, pred_now, zero_division=0):.4f}, "
        f"recall={recall_score(true_now, pred_now, zero_division=0):.4f}, "
        f"f1={f1_score(true_now, pred_now, zero_division=0):.4f}, "
        f"stage_size={int(pred_now.sum())}/{len(y_true_horizon)}"
    )

    # Stage 2: D1_7 among samples that survived stage 1.
    stage2_mask = ~pred_now
    if int(stage2_mask.sum()) > 0:
        pred_d1_7 = stage2_mask & (p_le7 >= thresholds["LE_7_DAYS"])
        true_d1_7_stage2 = y_true_horizon[stage2_mask] == 1
        print(
            "Stage 2 D1_7 gate: "
            f"precision={precision_score(true_d1_7_stage2, pred_d1_7[stage2_mask], zero_division=0):.4f}, "
            f"recall={recall_score(true_d1_7_stage2, pred_d1_7[stage2_mask], zero_division=0):.4f}, "
            f"f1={f1_score(true_d1_7_stage2, pred_d1_7[stage2_mask], zero_division=0):.4f}, "
            f"stage_size={int(stage2_mask.sum())}/{len(y_true_horizon)}"
        )
    else:
        print("Stage 2 D1_7 gate: no samples reached this stage.")

    # Stage 3: D8_30 among samples that survived stages 1 and 2.
    stage3_mask = stage2_mask & ~(p_le7 >= thresholds["LE_7_DAYS"])
    if int(stage3_mask.sum()) > 0:
        pred_d8_30 = stage3_mask & (p_le30 >= thresholds["LE_30_DAYS"])
        true_d8_30_stage3 = y_true_horizon[stage3_mask] == 2
        print(
            "Stage 3 D8_30 gate: "
            f"precision={precision_score(true_d8_30_stage3, pred_d8_30[stage3_mask], zero_division=0):.4f}, "
            f"recall={recall_score(true_d8_30_stage3, pred_d8_30[stage3_mask], zero_division=0):.4f}, "
            f"f1={f1_score(true_d8_30_stage3, pred_d8_30[stage3_mask], zero_division=0):.4f}, "
            f"stage_size={int(stage3_mask.sum())}/{len(y_true_horizon)}"
        )
    else:
        print("Stage 3 D8_30 gate: no samples reached this stage.")


def score_combined_predictions(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric == "priority_weighted_f1":
        class_f1 = f1_score(
            y_true,
            y_pred,
            labels=HORIZON_CLASS_LABELS,
            average=None,
            zero_division=0,
        )
        w = COMBINATION_CLASS_WEIGHTS
        return float(w[0] * class_f1[0] + w[1] * class_f1[1] + w[2] * class_f1[2] + w[3] * class_f1[3])
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "weighted_f1":
        return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    if metric == "macro_recall":
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported combination score metric: {metric}")


def tune_policy_thresholds(
    val_scores: dict[str, np.ndarray],
    y_true_val_horizon: np.ndarray,
    metric: str,
) -> tuple[dict[str, float], float]:
    best_thresholds_constrained: dict[str, float] | None = None
    best_score_constrained = -np.inf
    best_thresholds_unconstrained = DEFAULT_DEPLOY_THRESHOLDS.copy()
    best_score_unconstrained = -np.inf

    for t_now, t_7, t_30 in itertools.product(
        COMBINATION_THRESHOLD_GRID["NOW"],
        COMBINATION_THRESHOLD_GRID["LE_7_DAYS"],
        COMBINATION_THRESHOLD_GRID["LE_30_DAYS"],
    ):
        candidate = {
            "NOW": float(t_now),
            "LE_7_DAYS": float(t_7),
            "LE_30_DAYS": float(t_30),
        }
        y_pred = predict_horizon_by_policy(val_scores, candidate)
        score = score_combined_predictions(metric, y_true_val_horizon, y_pred)

        if score > best_score_unconstrained:
            best_score_unconstrained = score
            best_thresholds_unconstrained = candidate

        if ENFORCE_COMBINATION_RECALL_FLOORS:
            class_recalls = recall_score(
                y_true_val_horizon,
                y_pred,
                labels=HORIZON_CLASS_LABELS,
                average=None,
                zero_division=0,
            )
            is_feasible = all(
                class_recalls[i] >= COMBINATION_MIN_CLASS_RECALL[i]
                for i in HORIZON_CLASS_LABELS
            )
            if not is_feasible:
                continue

        if score > best_score_constrained:
            best_score_constrained = score
            best_thresholds_constrained = candidate

    if best_thresholds_constrained is not None:
        return best_thresholds_constrained, float(best_score_constrained)

    # Fall back to unconstrained best if recall floors are infeasible.
    print("[WARN] No threshold combination satisfied recall floors; using unconstrained best.")
    return best_thresholds_unconstrained, float(best_score_unconstrained)


def print_horizon_report(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n=== {title} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print("Class mapping: 0=NOW, 1=D1_7, 2=D8_30, 3=GT_30")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=HORIZON_CLASS_LABELS,
            target_names=HORIZON_CLASS_NAMES,
            zero_division=0,
            digits=3,
        )
    )
    cm = confusion_matrix(y_true, y_pred, labels=HORIZON_CLASS_LABELS)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


def format_deployment_thresholds(thresholds: dict[str, float]) -> str:
    return (
        "Suggested deployment confidence thresholds (fallback policy):\n"
        f"  NOW       : {thresholds['NOW']:.4f} ({thresholds['NOW']:.2%})\n"
        f"  LE_7_DAYS : {thresholds['LE_7_DAYS']:.4f} ({thresholds['LE_7_DAYS']:.2%})\n"
        f"  LE_30_DAYS: {thresholds['LE_30_DAYS']:.4f} ({thresholds['LE_30_DAYS']:.2%})\n"
        "  Note      : these are starting values for deployment; the cascade predictions in this\n"
        "              run still use the thresholds tuned per model on validation data."
    )


def print_confidence_combined_summary(y_true: np.ndarray, conf: np.ndarray, y_pred: np.ndarray) -> None:
    chosen_confidence = np.max(conf, axis=1)
    pred_counts = np.bincount(y_pred, minlength=len(HORIZON_CLASS_NAMES))
    true_counts = np.bincount(y_true, minlength=len(HORIZON_CLASS_NAMES))

    print("\n=== Combined Horizon Output From Confidence Scores ===")
    print("This uses normalized class confidences derived from the three cumulative binary models:")
    print("  NOW   = P(RUL <= 0)")
    print("  D1_7  = P(RUL <= 7)  - P(RUL <= 0)")
    print("  D8_30 = P(RUL <= 30) - P(RUL <= 7)")
    print("  GT_30 = 1 - P(RUL <= 30)")
    print_horizon_report("Confidence-Argmax Horizon Output", y_true, y_pred)
    print(
        "Confidence summary: "
        f"mean={float(np.mean(chosen_confidence)):.4f}, "
        f"median={float(np.median(chosen_confidence)):.4f}, "
        f"min={float(np.min(chosen_confidence)):.4f}, "
        f"p10={float(np.quantile(chosen_confidence, 0.10)):.4f}, "
        f"p90={float(np.quantile(chosen_confidence, 0.90)):.4f}"
    )
    print("Predicted class distribution:")
    for idx, name in enumerate(HORIZON_CLASS_NAMES):
        print(f"  {name:<6}: {int(pred_counts[idx]):>6d} ({pred_counts[idx] / len(y_pred):6.2%})")
    print("True class distribution:")
    for idx, name in enumerate(HORIZON_CLASS_NAMES):
        print(f"  {name:<6}: {int(true_counts[idx]):>6d} ({true_counts[idx] / len(y_true):6.2%})")


def _horizon_metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _plot_confusion(ax, cm: np.ndarray, title: str, weighted: bool = True) -> None:
    display = cm.astype(float)
    if weighted:
        row_sum = display.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        display = display / row_sum

    im = ax.imshow(display, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0 if weighted else None)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(HORIZON_CLASS_NAMES)), HORIZON_CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(HORIZON_CLASS_NAMES)), HORIZON_CLASS_NAMES)

    threshold = display.max() / 2.0 if display.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            label = f"{display[i, j]:.2%}" if weighted else f"{int(cm[i, j])}"
            ax.text(j, i, label, ha="center", va="center", color=color)

    # Add a compact colorbar next to each confusion matrix.
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_combined_result_plots(
    y_true: np.ndarray,
    y_pred_confidence: np.ndarray,
    y_pred_policy: np.ndarray,
    output_dir: str,
    show_plot: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for combined result plots. Install with `pip install matplotlib`."
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_conf = _horizon_metric_summary(y_true, y_pred_confidence)
    metrics_policy = _horizon_metric_summary(y_true, y_pred_policy)

    metric_labels = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    conf_vals = [metrics_conf[m] for m in metric_labels]
    policy_vals = [metrics_policy[m] for m in metric_labels]

    cm_conf = confusion_matrix(y_true, y_pred_confidence, labels=HORIZON_CLASS_LABELS)
    cm_policy = confusion_matrix(y_true, y_pred_policy, labels=HORIZON_CLASS_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    x = np.arange(len(metric_labels))
    width = 0.38
    axes[0].bar(x - width / 2, conf_vals, width=width, label="Confidence argmax", color="#1f77b4")
    axes[0].bar(x + width / 2, policy_vals, width=width, label="Deploy policy", color="#ff7f0e")
    axes[0].set_xticks(x, [m.replace("_", "\n") for m in metric_labels])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Combined Horizon Metrics")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", alpha=0.3)

    _plot_confusion(axes[1], cm_conf, "Weighted Confusion Matrix: Confidence argmax", weighted=True)
    _plot_confusion(axes[2], cm_policy, "Weighted Confusion Matrix: Deploy policy", weighted=True)

    fig.suptitle("Combined Horizon Performance Overview", fontsize=14)
    fig.tight_layout()

    out_path = out_dir / "combined_horizon_results.png"
    fig.savefig(out_path, dpi=160)

    # Also save each panel separately for easier report inclusion.
    metric_fig, metric_ax = plt.subplots(figsize=(9, 5))
    metric_ax.bar(x - width / 2, conf_vals, width=width, label="Confidence argmax", color="#1f77b4")
    metric_ax.bar(x + width / 2, policy_vals, width=width, label="Deploy policy", color="#ff7f0e")
    metric_ax.set_xticks(x, [m.replace("_", "\n") for m in metric_labels])
    metric_ax.set_ylim(0.0, 1.0)
    metric_ax.set_title("Combined Horizon Metrics")
    metric_ax.set_ylabel("Score")
    metric_ax.legend(loc="lower right")
    metric_ax.grid(axis="y", alpha=0.3)
    metric_fig.tight_layout()
    metric_fig.savefig(out_dir / "combined_horizon_metrics.png", dpi=160)
    plt.close(metric_fig)

    conf_fig, conf_ax = plt.subplots(figsize=(7, 6))
    _plot_confusion(conf_ax, cm_conf, "Confusion Matrix: Confidence argmax")
    conf_fig.tight_layout()
    conf_fig.savefig(out_dir / "combined_confidence_confusion.png", dpi=160)
    plt.close(conf_fig)

    policy_fig, policy_ax = plt.subplots(figsize=(7, 6))
    _plot_confusion(policy_ax, cm_policy, "Confusion Matrix: Deploy policy")
    policy_fig.tight_layout()
    policy_fig.savefig(out_dir / "combined_policy_confusion.png", dpi=160)
    plt.close(policy_fig)

    if show_plot:
        plt.show()
    plt.close(fig)

    print(f"\nSaved combined results plot to: {out_path.resolve()}")


def _plot_binary_confusion(ax, cm: np.ndarray, title: str, weighted: bool = True) -> None:
    display = cm.astype(float)
    if weighted:
        row_sum = display.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        display = display / row_sum

    im = ax.imshow(display, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0 if weighted else None)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Neg", "Pos"])
    ax.set_yticks([0, 1], ["Neg", "Pos"])

    threshold = display.max() / 2.0 if display.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            label = f"{display[i, j]:.2%}" if weighted else f"{int(cm[i, j])}"
            ax.text(j, i, label, ha="center", va="center", color=color)

    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_binary_model_report_plots(
    results: list[BinaryResult],
    y_true_by_task: dict[str, np.ndarray],
    y_score_by_task: dict[str, np.ndarray],
    output_dir: str,
    show_plot: bool,
) -> None:
    """Save a compact report with per-model binary confusion matrices and PR curves."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for report plots. Install with `pip install matplotlib`."
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_map = {r.task: r for r in results}
    tasks = [r.task for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Per-Model Report: Confusion Matrices and Precision-Recall Curves", fontsize=15)

    for idx, task in enumerate(tasks):
        y_true = y_true_by_task[task]
        y_score = y_score_by_task[task]
        threshold = result_map[task].threshold
        y_pred = (y_score >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        _plot_binary_confusion(axes[0, idx], cm, f"{task} weighted confusion")

        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        axes[1, idx].plot(recall, precision, color="#1f77b4", lw=2, label=f"AP={ap:.3f}")
        axes[1, idx].scatter(
            [result_map[task].recall],
            [result_map[task].precision],
            s=50,
            color="#d62728",
            zorder=3,
            label=f"threshold={threshold:.4f}",
        )
        axes[1, idx].set_title(f"{task} precision-recall")
        axes[1, idx].set_xlabel("Recall")
        axes[1, idx].set_ylabel("Precision")
        axes[1, idx].set_xlim(0.0, 1.0)
        axes[1, idx].set_ylim(0.0, 1.05)
        axes[1, idx].grid(alpha=0.3)
        axes[1, idx].legend(loc="lower left", fontsize=9)

        # Add a small annotation with key metrics.
        axes[1, idx].text(
            0.02,
            0.05,
            f"F1={result_map[task].f1:.3f}\nRecall={result_map[task].recall:.3f}\nPrecision={result_map[task].precision:.3f}",
            transform=axes[1, idx].transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )

        # Save separate per-model figures as well.
        single_conf_fig, single_conf_ax = plt.subplots(figsize=(5.5, 5))
        _plot_binary_confusion(single_conf_ax, cm, f"{task} weighted confusion")
        single_conf_fig.tight_layout()
        single_conf_fig.savefig(out_dir / f"{task.lower()}_confusion.png", dpi=160)
        plt.close(single_conf_fig)

        single_pr_fig, single_pr_ax = plt.subplots(figsize=(6, 5))
        single_pr_ax.plot(recall, precision, color="#1f77b4", lw=2, label=f"AP={ap:.3f}")
        single_pr_ax.scatter(
            [result_map[task].recall],
            [result_map[task].precision],
            s=50,
            color="#d62728",
            zorder=3,
            label=f"threshold={threshold:.4f}",
        )
        single_pr_ax.set_title(f"{task} precision-recall")
        single_pr_ax.set_xlabel("Recall")
        single_pr_ax.set_ylabel("Precision")
        single_pr_ax.set_xlim(0.0, 1.0)
        single_pr_ax.set_ylim(0.0, 1.05)
        single_pr_ax.grid(alpha=0.3)
        single_pr_ax.legend(loc="lower left", fontsize=9)
        single_pr_ax.text(
            0.02,
            0.05,
            f"F1={result_map[task].f1:.3f}\nRecall={result_map[task].recall:.3f}\nPrecision={result_map[task].precision:.3f}",
            transform=single_pr_ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )
        single_pr_fig.tight_layout()
        single_pr_fig.savefig(out_dir / f"{task.lower()}_precision_recall.png", dpi=160)
        plt.close(single_pr_fig)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "binary_model_report.png"
    fig.savefig(out_path, dpi=160)
    if show_plot:
        plt.show()
    plt.close(fig)

    print(f"Saved binary model report plot to: {out_path.resolve()}")


def build_xgb_pipeline(pre: ColumnTransformer, params: dict) -> Pipeline:
    xgb = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=DEFAULT_XGB_SUBSAMPLE,
        colsample_bytree=DEFAULT_XGB_COLSAMPLE_BYTREE,
        reg_lambda=DEFAULT_XGB_REG_LAMBDA,
        random_state=DEFAULT_XGB_RANDOM_STATE,
        n_jobs=DEFAULT_XGB_N_JOBS,
        eval_metric=DEFAULT_XGB_EVAL_METRIC,
    )

    steps: list[tuple[str, object]] = [("pre", pre)]
    if USE_PCA:
        steps.append(("pca", PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)))
    steps.append(("clf", xgb))
    return Pipeline(steps=steps)


def main() -> None:
    if XGBClassifier is None:
        raise ModuleNotFoundError("xgboost is required. Install with `pip install xgboost`.")

    parser = argparse.ArgumentParser(description="Train 3 separate XGBoost horizons + combined output (time split)")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument("--beta", type=float, default=GLOBAL_BETA)
    parser.add_argument(
        "--save-combined-plots",
        action="store_true",
        default=DEFAULT_SAVE_COMBINED_PLOTS,
        help="Save combined metrics/confusion visualizations.",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default=DEFAULT_PLOT_OUTPUT_DIR,
        help="Directory for combined result plots.",
    )
    parser.add_argument(
        "--show-combined-plots",
        action="store_true",
        default=DEFAULT_SHOW_COMBINED_PLOTS,
        help="Display combined result plots interactively.",
    )
    parser.add_argument(
        "--auto-tune-combination",
        action="store_true",
        default=AUTO_TUNE_COMBINATION,
        help="Tune deploy thresholds on validation for better combined performance.",
    )
    parser.add_argument(
        "--combination-metric",
        type=str,
        choices=COMBINATION_SCORE_CHOICES,
        default=COMBINATION_SCORE_METRIC,
        help="Objective used for combined threshold tuning.",
    )
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
        train_df = train_df.sample(n=args.max_train_rows, random_state=RANDOM_STATE).sort_values("DATE").reset_index(drop=True)

    feature_cols = list(DEFAULT_BASE_FEATURE_COLS) + sensor_cols + [f"{s}_lag1" for s in sensor_cols] + [f"{s}_roll7_mean" for s in sensor_cols]

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
                        ("imputer", SimpleImputer(strategy=CATEGORICAL_IMPUTER_STRATEGY)),
                        ("onehot", OneHotEncoder(handle_unknown=ONEHOT_HANDLE_UNKNOWN)),
                    ]
                ),
                categorical_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy=NUMERIC_IMPUTER_STRATEGY)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )

    task_definitions = list(DEFAULT_TASK_DEFINITIONS)
    results: list[BinaryResult] = []
    model_val_scores: dict[str, np.ndarray] = {}
    model_test_scores: dict[str, np.ndarray] = {}

    print("=== XGBoost Three-Horizon Training (Time Split) ===")
    print(f"Rows: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    print(f"Bins: {BIN_MAX_DAYS}")
    print(f"PCA enabled: {USE_PCA} (n_components={PCA_N_COMPONENTS})")
    print(format_deployment_thresholds(DEFAULT_DEPLOY_THRESHOLDS))

    for task_name, max_days in task_definitions:
        if task_name not in XGB_PARAMS_BY_TASK:
            raise KeyError(f"Missing params for task: {task_name}")

        y_train = (train_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        y_val = (val_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        y_test = (test_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)

        params = XGB_PARAMS_BY_TASK[task_name]

        pipe = build_xgb_pipeline(pre=pre, params=params)
        pipe.fit(X_train, y_train)

        val_scores = pipe.predict_proba(X_val)[:, 1]
        test_scores = pipe.predict_proba(X_test)[:, 1]
        beta_task = TASK_BETA.get(task_name, args.beta) if USE_TASK_SPECIFIC_BETA else args.beta

        threshold = tune_threshold(y_val, val_scores, beta=beta_task)
        metrics = evaluate_binary(y_test, test_scores, threshold=threshold, beta=beta_task)
        model_val_scores[task_name] = val_scores
        model_test_scores[task_name] = test_scores

        results.append(
            BinaryResult(
                task=task_name,
                threshold=threshold,
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                fbeta=metrics["fbeta"],
                pr_auc=metrics["pr_auc"],
                roc_auc=metrics["roc_auc"],
                alert_rate=metrics["alert_rate"],
                positives_train=int(np.sum(y_train)),
                positives_val=int(np.sum(y_val)),
                positives_test=int(np.sum(y_test)),
            )
        )

        print(f"\nTask {task_name}: y=1 if RUL_DAYS <= {max_days}")
        print(f"  positives train/val/test: {int(np.sum(y_train))}/{int(np.sum(y_val))}/{int(np.sum(y_test))}")
        print(
            "  test metrics: "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"fbeta={metrics['fbeta']:.4f}, pr_auc={metrics['pr_auc']:.4f}, "
            f"threshold={threshold:.4f}"
        )
        y_test_pred = (test_scores >= threshold).astype(int)
        cm_binary = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
        print("  confusion matrix (rows=true, cols=pred, labels=[0,1]):")
        print(cm_binary)

    # Combined multiclass output from the three cumulative binary models.
    y_true_horizon = build_true_horizon_class(test_df["RUL_DAYS"].to_numpy())
    conf = build_horizon_confidence_table(model_test_scores)
    y_pred_horizon = np.argmax(conf, axis=1)
    y_pred_confidence = np.max(conf, axis=1)

    print_confidence_combined_summary(y_true_horizon, conf, y_pred_horizon)
    print(f"Mean chosen-class confidence: {float(np.mean(y_pred_confidence)):.4f}")

    cascade_thresholds = {r.task: r.threshold for r in results}
    print(f"\nCascade thresholds from per-model tuning: {cascade_thresholds}")

    y_pred_cascade = predict_horizon_by_cascade(model_test_scores, cascade_thresholds)
    print_horizon_report(
        "Cascade Horizon Output (NOW -> D1_7 -> D8_30 -> GT_30)",
        y_true_horizon,
        y_pred_cascade,
    )
    print_cascade_stage_metrics(y_true_horizon, model_test_scores, cascade_thresholds)

    if args.save_combined_plots:
        save_combined_result_plots(
            y_true=y_true_horizon,
            y_pred_confidence=y_pred_horizon,
            y_pred_policy=y_pred_cascade,
            output_dir=args.plot_output_dir,
            show_plot=args.show_combined_plots,
        )

    out = pd.DataFrame([r.__dict__ for r in results])
    print("\n=== Per-task summary ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.save_combined_plots:
        y_true_by_task = {
            task_name: (test_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
            for task_name, max_days in task_definitions
        }
        save_binary_model_report_plots(
            results=results,
            y_true_by_task=y_true_by_task,
            y_score_by_task=model_test_scores,
            output_dir=args.plot_output_dir,
            show_plot=args.show_combined_plots,
        )


if __name__ == "__main__":
    main()
