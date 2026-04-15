"""
Time-split one-vs-rest horizon prediction for near-failure detection.

Tasks:
- NOW:        failure today (RUL_DAYS == 0)
- LE_7_DAYS:  failure within 7 days (RUL_DAYS <= 7)
- LE_30_DAYS: failure within 30 days (RUL_DAYS <= 30)

For each task:
- Build target from same train/val/test time split
- Train baseline + selected model (config at top)
- Tune threshold on validation via F-beta
- Evaluate on test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.decomposition import PCA
from tqdm import tqdm
import itertools
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:
    LGBMClassifier = None


CSV_PATH = "equipment_failure_data_1.csv"
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_MAX_TRAIN_ROWS = 0
DEFAULT_BETA = 2.0
DEFAULT_PLOT_PR_CURVES = True
DEFAULT_PLOT_OUTPUT_DIR = "plots"
DEFAULT_SHOW_PR_CURVES = True

# Tunable feature/config settings.
DEFAULT_SENSOR_COLS = ("S5", "S8", "S13", "S15", "S16", "S17", "S18", "S19")
DEFAULT_ROLLING_WINDOW = 7
DEFAULT_ROLLING_MIN_PERIODS = 2

# Tunable model selection/settings.
# Allowed values: "logreg", "rf", "hgb", "xgb", "lgbm", "calibrated_best"
DEFAULT_MODEL_TYPE = "xgb"
MODEL_TYPE_CHOICES = ("logreg", "rf", "hgb", "xgb", "lgbm", "calibrated_best")

# Logistic Regression settings
DEFAULT_LOGREG_NAME = "LogReg"
DEFAULT_LOGREG_MAX_ITER = 15000
DEFAULT_LOGREG_CLASS_WEIGHT = "balanced"
DEFAULT_LOGREG_SOLVER = "liblinear"
DEFAULT_LOGREG_RANDOM_STATE = 42

# Random Forest settings
DEFAULT_RF_NAME = "RF"
DEFAULT_RF_N_ESTIMATORS = 500
DEFAULT_RF_CLASS_WEIGHT = "balanced_subsample"
DEFAULT_RF_RANDOM_STATE = 42
DEFAULT_RF_N_JOBS = -1

# HistGradientBoosting settings
DEFAULT_HGB_NAME = "HGB"
DEFAULT_HGB_MAX_ITER = 350
DEFAULT_HGB_LEARNING_RATE = 0.06
DEFAULT_HGB_MAX_DEPTH = 6
DEFAULT_HGB_MIN_SAMPLES_LEAF = 35
DEFAULT_HGB_L2_REGULARIZATION = 0.05
DEFAULT_HGB_RANDOM_STATE = 42

# XGBoost settings (requires xgboost package)
DEFAULT_XGB_NAME = "XGBoost"
DEFAULT_XGB_N_ESTIMATORS = 261 #350 #261, 69, 703
DEFAULT_XGB_MAX_DEPTH = 10 #6 #11, 15, 15
DEFAULT_XGB_LEARNING_RATE = 0.002 #0.05 #0.005, 0.11499999, 0.06
DEFAULT_XGB_SUBSAMPLE = 0.85
DEFAULT_XGB_COLSAMPLE_BYTREE = 0.85
DEFAULT_XGB_REG_LAMBDA = 1.0
DEFAULT_XGB_RANDOM_STATE = 42
DEFAULT_XGB_N_JOBS = -1
#NOW best (0.3) parameters: DEFAULT_XGB_N_ESTIMATORS = 261, DEFAULT_XGB_MAX_DEPTH = 15, DEFAULT_XGB_LEARNING_RATE = 0.005
#LE_7_DAYS best (0.2) parameters: DEFAULT_XGB_N_ESTIMATORS = 261, DEFAULT_XGB_MAX_DEPTH = 10, DEFAULT_XGB_LEARNING_RATE = 0.002
#LE_30_DAYS best (0.27) parameters: DEFAULT_XGB_N_ESTIMATORS = 703, DEFAULT_XGB_MAX_DEPTH = 15, DEFAULT_XGB_LEARNING_RATE = 0.06

# LightGBM settings (requires lightgbm package)
DEFAULT_LGBM_NAME = "LightGBM"
DEFAULT_LGBM_N_ESTIMATORS = 400
DEFAULT_LGBM_MAX_DEPTH = -1
DEFAULT_LGBM_LEARNING_RATE = 0.05
DEFAULT_LGBM_NUM_LEAVES = 63
DEFAULT_LGBM_SUBSAMPLE = 0.9
DEFAULT_LGBM_COLSAMPLE_BYTREE = 0.9
DEFAULT_LGBM_RANDOM_STATE = 42
DEFAULT_LGBM_N_JOBS = -1

# Calibrated best-model settings.
DEFAULT_CALIBRATED_BEST_NAME = "CalibratedBest"
DEFAULT_CALIBRATED_BEST_CANDIDATES = ("logreg", "rf", "hgb", "xgb", "lgbm")
DEFAULT_CALIBRATION_METHOD = "sigmoid"
DEFAULT_CALIBRATION_CV = 3

# One-vs-rest task definitions: (task_name, max_rul_days_inclusive)
# For NOW, use 0 (exactly failure day).
DEFAULT_TASK_DEFINITIONS = (
    ("NOW", 0),
    ("LE_7_DAYS", 7),
    ("LE_30_DAYS", 30),
)

# Per-task tuning aggressiveness.
# beta > 1: recall-heavy (more aggressive), beta < 1: precision-heavy (more conservative)
DEFAULT_USE_TASK_SPECIFIC_BETA = True
DEFAULT_TASK_BETA = {
    "NOW": 3.0,
    "LE_7_DAYS": 1.5,
    "LE_30_DAYS": 0.8,
}

# Deployment policy thresholds for priority rule NOW -> LE_7_DAYS -> LE_30_DAYS -> GT_30.
DEFAULT_DEPLOY_THRESHOLDS = {
    "NOW": 0.1,
    "LE_7_DAYS": 0.05,
    "LE_30_DAYS": 0.02,
}

# Static dataset columns expected in the CSV.
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

HORIZON_CLASS_LABELS = [0, 1, 2, 3]
HORIZON_CLASS_NAMES = ["NOW", "D1_7", "D8_30", "GT_30"]


@dataclass
class BinaryResult:
    task: str
    model: str
    threshold: float
    precision: float
    recall: float
    f1: float
    fbeta: float
    pr_auc: float
    roc_auc: float
    alert_rate: float



@dataclass
class CurvePayload:
    task: str
    model_name: str
    y_true: np.ndarray
    baseline_scores: np.ndarray
    model_scores: np.ndarray
    y_pred: np.ndarray


def _is_model_available(model_type: str) -> bool:
    if model_type == "xgb":
        return XGBClassifier is not None
    if model_type == "lgbm":
        return LGBMClassifier is not None
    return model_type in {"logreg", "rf", "hgb", "calibrated_best"}


def _create_base_estimator(model_type: str):
    if model_type == "logreg":
        return DEFAULT_LOGREG_NAME, LogisticRegression(
            max_iter=DEFAULT_LOGREG_MAX_ITER,
            class_weight=DEFAULT_LOGREG_CLASS_WEIGHT,
            solver=DEFAULT_LOGREG_SOLVER,
            random_state=DEFAULT_LOGREG_RANDOM_STATE,
        )

    if model_type == "rf":
        return DEFAULT_RF_NAME, RandomForestClassifier(
            n_estimators=DEFAULT_RF_N_ESTIMATORS,
            class_weight=DEFAULT_RF_CLASS_WEIGHT,
            random_state=DEFAULT_RF_RANDOM_STATE,
            n_jobs=DEFAULT_RF_N_JOBS,
        )

    if model_type == "hgb":
        return DEFAULT_HGB_NAME, HistGradientBoostingClassifier(
            max_iter=DEFAULT_HGB_MAX_ITER,
            learning_rate=DEFAULT_HGB_LEARNING_RATE,
            max_depth=DEFAULT_HGB_MAX_DEPTH,
            min_samples_leaf=DEFAULT_HGB_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_HGB_L2_REGULARIZATION,
            random_state=DEFAULT_HGB_RANDOM_STATE,
        )

    if model_type == "xgb":
        if XGBClassifier is None:
            raise ModuleNotFoundError("Model type 'xgb' requires package xgboost. Install with `pip install xgboost`.")
        return DEFAULT_XGB_NAME, XGBClassifier(
            n_estimators=DEFAULT_XGB_N_ESTIMATORS,
            max_depth=DEFAULT_XGB_MAX_DEPTH,
            learning_rate=DEFAULT_XGB_LEARNING_RATE,
            subsample=DEFAULT_XGB_SUBSAMPLE,
            colsample_bytree=DEFAULT_XGB_COLSAMPLE_BYTREE,
            reg_lambda=DEFAULT_XGB_REG_LAMBDA,
            random_state=DEFAULT_XGB_RANDOM_STATE,
            n_jobs=DEFAULT_XGB_N_JOBS,
            eval_metric="logloss",
        )

    if model_type == "lgbm":
        if LGBMClassifier is None:
            raise ModuleNotFoundError("Model type 'lgbm' requires package lightgbm. Install with `pip install lightgbm`.")
        return DEFAULT_LGBM_NAME, LGBMClassifier(
            n_estimators=DEFAULT_LGBM_N_ESTIMATORS,
            max_depth=DEFAULT_LGBM_MAX_DEPTH,
            learning_rate=DEFAULT_LGBM_LEARNING_RATE,
            num_leaves=DEFAULT_LGBM_NUM_LEAVES,
            subsample=DEFAULT_LGBM_SUBSAMPLE,
            colsample_bytree=DEFAULT_LGBM_COLSAMPLE_BYTREE,
            random_state=DEFAULT_LGBM_RANDOM_STATE,
            n_jobs=DEFAULT_LGBM_N_JOBS,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def fit_selected_model(
    model_type: str,
    pre: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
) -> tuple[str, np.ndarray, np.ndarray]:
    if model_type != "calibrated_best":
        model_name, clf = _create_base_estimator(model_type)
        pipe = Pipeline(steps=[("pre", pre), ("pca", PCA(n_components=0.95, random_state=42)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        return model_name, pipe.predict_proba(X_val)[:, 1], pipe.predict_proba(X_test)[:, 1]

    candidates = [m for m in DEFAULT_CALIBRATED_BEST_CANDIDATES if _is_model_available(m)]
    if not candidates:
        raise ValueError(
            "No available candidates for calibrated_best. "
            "Install xgboost/lightgbm or keep at least one of logreg/rf/hgb in candidates."
        )

    best_model_type = None
    best_ap = -np.inf
    for candidate in candidates:
        _, base_est = _create_base_estimator(candidate)
        base_pipe = Pipeline(steps=[("pre", pre), ("pca", PCA(n_components=0.95, random_state=42)), ("clf", base_est)])
        base_pipe.fit(X_train, y_train)
        val_scores = base_pipe.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, val_scores)
        if ap > best_ap:
            best_ap = ap
            best_model_type = candidate

    if best_model_type is None:
        raise RuntimeError("Could not determine best model for calibration.")

    best_name, best_estimator = _create_base_estimator(best_model_type)
    calibrated = CalibratedClassifierCV(
        estimator=best_estimator,
        method=DEFAULT_CALIBRATION_METHOD,
        cv=DEFAULT_CALIBRATION_CV,
    )
    calibrated_pipe = Pipeline(steps=[("pre", pre), ("pca", PCA(n_components=0.95, random_state=42)), ("clf", calibrated)])
    calibrated_pipe.fit(X_train, y_train)
    display_name = f"{DEFAULT_CALIBRATED_BEST_NAME}({best_name})"
    return display_name, calibrated_pipe.predict_proba(X_val)[:, 1], calibrated_pipe.predict_proba(X_test)[:, 1]


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


def evaluate_binary(task: str, model: str, y_true: np.ndarray, y_score: np.ndarray, threshold: float, beta: float) -> BinaryResult:
    y_pred = (y_score >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fbeta = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_score))
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = float("nan")

    return BinaryResult(
        task=task,
        model=model,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        fbeta=fbeta,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        alert_rate=float(y_pred.mean()),
    )


def save_precision_recall_curves(curves: list[CurvePayload], output_dir: str, show_plots: bool) -> None:
    # Local import keeps plotting optional for users who do not have matplotlib installed.
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for --plot-pr-curves. "
            "Install it in your active environment, e.g. `pip install matplotlib`."
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for curve in curves:
        p_b, r_b, _ = precision_recall_curve(curve.y_true, curve.baseline_scores)
        p_m, r_m, _ = precision_recall_curve(curve.y_true, curve.model_scores)

        ap_b = average_precision_score(curve.y_true, curve.baseline_scores)
        ap_m = average_precision_score(curve.y_true, curve.model_scores)

        plt.figure(figsize=(7, 5))
        plt.plot(r_b, p_b, label=f"Baseline AP={ap_b:.3f}")
        plt.plot(r_m, p_m, label=f"{curve.model_name} AP={ap_m:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve: {curve.task}")
        plt.legend()
        plt.grid(alpha=0.3)

        out_path = out_dir / f"pr_curve_{curve.task.lower()}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        if show_plots:
            plt.show()
        plt.close()

        cm = confusion_matrix(curve.y_true, curve.y_pred)

        # Normalize (choose one)
        cm_norm = cm / cm.sum()  # global normalization
        # OR:
        # cm_norm = cm / cm.sum(axis=1, keepdims=True)  # row-wise

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"]
        )

        plt.title(f"{curve.task}: Confusion Matrix (Normalized)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        out_path_cm = out_dir / f"confusion_matrix_{curve.task.lower()}.png"
        plt.tight_layout()
        plt.savefig(out_path_cm, dpi=150)

        if show_plots:
            plt.show()
        plt.close()

    print(f"\nSaved PR curves and Confusion Matrix to: {out_dir.resolve()}")


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
    """Convert three cumulative binary probabilities into four horizon confidences.

    Inputs expected in model_scores:
    - NOW: P(RUL <= 0)
    - LE_7_DAYS: P(RUL <= 7)
    - LE_30_DAYS: P(RUL <= 30)
    """
    p_now_raw = model_scores["NOW"]
    p_le7_raw = model_scores["LE_7_DAYS"]
    p_le30_raw = model_scores["LE_30_DAYS"]

    # Enforce monotonic cumulative probabilities: NOW <= LE_7 <= LE_30.
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
    conf = conf / row_sum
    return conf


def predict_horizon_by_policy(model_scores: dict[str, np.ndarray], thresholds: dict[str, float]) -> np.ndarray:
    """Priority rule prediction with explicit thresholds.

    0 (NOW) if p_now >= t_now
    1 (D1_7) if p_le7 >= t_7
    2 (D8_30) if p_le30 >= t_30
    3 (GT_30) otherwise
    """
    p_now = model_scores["NOW"]
    p_le7 = model_scores["LE_7_DAYS"]
    p_le30 = model_scores["LE_30_DAYS"]

    pred = np.full(shape=p_now.shape[0], fill_value=3, dtype=int)
    pred[p_le30 >= thresholds["LE_30_DAYS"]] = 2
    pred[p_le7 >= thresholds["LE_7_DAYS"]] = 1
    pred[p_now >= thresholds["NOW"]] = 0
    return pred


def print_horizon_report(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n=== {title} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
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


def main() -> None:
    global DEFAULT_XGB_LEARNING_RATE, DEFAULT_XGB_N_ESTIMATORS, DEFAULT_XGB_MAX_DEPTH
    parser = argparse.ArgumentParser(description="Time-split one-vs-rest horizon tasks")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to CSV")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=MODEL_TYPE_CHOICES,
        help="Model backend to use",
    )
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="F-beta used for threshold tuning")
    parser.add_argument(
        "--plot-pr-curves",
        action="store_true",
        default=DEFAULT_PLOT_PR_CURVES,
        help="Save precision-recall curve PNGs for each task",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default=DEFAULT_PLOT_OUTPUT_DIR,
        help="Directory where PR curve images are saved",
    )
    parser.add_argument(
        "--show-pr-curves",
        action="store_true",
        default=DEFAULT_SHOW_PR_CURVES,
        help="Show PR curves interactively (also saves PNGs)",
    )
    args = parser.parse_args()

    selected_model_type = args.model_type
    if selected_model_type in {"xgb", "lgbm"} and not _is_model_available(selected_model_type):
        install_hint = "xgboost" if selected_model_type == "xgb" else "lightgbm"
        raise ModuleNotFoundError(
            f"Model type '{selected_model_type}' selected but package is not installed. "
            f"Install with `pip install {install_hint}`."
        )

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

    task_definitions = list(DEFAULT_TASK_DEFINITIONS)

    results: list[BinaryResult] = []
    curve_payloads: list[CurvePayload] = []
    model_test_scores: dict[str, np.ndarray] = {}
    model_display_name_for_task: dict[str, str] = {}
    estimator_range = np.linspace(50, 800, 40)
    depth_range = np.linspace(2, 20, 5)
    learning_range = np.linspace(0.5, 5e-3, 10)
    param_grid = list(itertools.product(learning_range, estimator_range, depth_range))
    param_grid = random.sample(param_grid, 300)  # instead of 2000

    for task_name, max_days in task_definitions:
        #"""
        #Optimization algorithm
        #"""
        # best_val_score = -1
        # best_lr = None
        # best_er = None
        # best_dr = None
        # print(f"\nTuning for task: {task_name}")
        # tqdm_bar = tqdm(param_grid, desc=f"{task_name} grid search")
        # for lr, er, dr in tqdm_bar:
        #     DEFAULT_XGB_LEARNING_RATE = lr
        #     DEFAULT_XGB_N_ESTIMATORS = int(er)
        #     DEFAULT_XGB_MAX_DEPTH = int(dr)
        #
        #     y_train = (train_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        #     y_val = (val_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        #     y_test = (test_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        #
        #     beta_task = DEFAULT_TASK_BETA.get(task_name, args.beta) if DEFAULT_USE_TASK_SPECIFIC_BETA else args.beta
        #
        #     # Model
        #     model_display_name, val_score_model, test_score_model = fit_selected_model(
        #         selected_model_type,
        #         pre,
        #         X_train,
        #         y_train,
        #         X_val,
        #         y_val,
        #         X_test,
        #     )
        #
        #     thr_val = tune_threshold(y_val, val_score_model, beta=beta_task)
        #     y_val_pred = (val_score_model >= thr_val).astype(int)
        #     f1_val = fbeta_score(y_val, y_val_pred, beta=beta_task, zero_division=0)
        #
        #     if f1_val > best_val_score:
        #         best_val_score = f1_val
        #         best_lr = lr
        #         best_er = int(er)
        #         best_dr = int(dr)
        #     tqdm_bar.set_postfix({
        #     "best_f1": f"{best_val_score:.4f}",
        #     "lr": f"{lr:.4f}",
        #     "n": int(er),
        #     "depth": int(dr)})
        #
        # DEFAULT_XGB_LEARNING_RATE = best_lr
        # DEFAULT_XGB_N_ESTIMATORS = best_er
        # DEFAULT_XGB_MAX_DEPTH = best_dr
        # print("Task: ",task_name,"\nBest validation f1: ",best_val_score,"\nDEFAULT_XGB_LEARNING_RATE: ", DEFAULT_XGB_LEARNING_RATE, "\nDEFAULT_XGB_N_ESTIMATORS: ", DEFAULT_XGB_N_ESTIMATORS, "\nDEFAULT_XGB_MAX_DEPTH: ", DEFAULT_XGB_MAX_DEPTH)

        y_train = (train_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        y_val = (val_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        y_test = (test_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)
        beta_task = DEFAULT_TASK_BETA.get(task_name, args.beta) if DEFAULT_USE_TASK_SPECIFIC_BETA else args.beta

        # Baseline risk: global positive rate from training data only.
        global_rate = float(np.mean(y_train))
        val_score_base = np.full(shape=len(X_val), fill_value=global_rate, dtype=float)
        test_score_base = np.full(shape=len(X_test), fill_value=global_rate, dtype=float)
        thr_base = tune_threshold(y_val, val_score_base, beta=beta_task)
        results.append(
            evaluate_binary(task_name, "Baseline(global rate)", y_test, test_score_base, thr_base, beta=beta_task)
        )

        # Model
        model_display_name, val_score_model, test_score_model = fit_selected_model(
            selected_model_type,
            pre,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
        )

        model_test_scores[task_name] = test_score_model
        model_display_name_for_task[task_name] = model_display_name
        thr_model = tune_threshold(y_val, val_score_model, beta=beta_task)
        results.append(
            evaluate_binary(task_name, model_display_name, y_test, test_score_model, thr_model, beta=beta_task)
        )
        y_pred_test = (test_score_model >= thr_model).astype(int)
        curve_payloads.append(
            CurvePayload(
                task=task_name,
                model_name=model_display_name,
                y_true=y_test,
                baseline_scores=test_score_base,
                model_scores=test_score_model,
                y_pred=y_pred_test
            )
        )
    out = pd.DataFrame([r.__dict__ for r in results])

    print("=== Time-Split One-vs-Rest Horizon Tasks ===")
    print(f"Rows: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    print(f"Selected model type: {selected_model_type}")
    print(f"Threshold tuning objective: F-beta with beta={args.beta}")
    if DEFAULT_USE_TASK_SPECIFIC_BETA:
        print(f"Task-specific beta tuning enabled: {DEFAULT_TASK_BETA}")
    else:
        print("Task-specific beta tuning disabled; using global beta for all tasks")
    print(f"Deployment policy thresholds: {DEFAULT_DEPLOY_THRESHOLDS}")

    for task_name, max_days in task_definitions:
        subset = out[out["task"] == task_name].sort_values("fbeta", ascending=False)
        pos_rate = float(np.mean((test_df["RUL_DAYS"].to_numpy() <= max_days).astype(int)))
        print(f"\nTask: {task_name} (RUL <= {max_days}, test positive rate={pos_rate:.4f})")
        print(f"Model used for task: {model_display_name_for_task.get(task_name, 'n/a')}")
        print(subset.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.plot_pr_curves:
        save_precision_recall_curves(curve_payloads, args.plot_output_dir, show_plots=args.show_pr_curves)

    # Combined output from horizon confidences (argmax over 4 horizon classes).
    y_true_horizon = build_true_horizon_class(test_df["RUL_DAYS"].to_numpy())
    conf = build_horizon_confidence_table(model_test_scores)
    y_pred_horizon = np.argmax(conf, axis=1)
    y_pred_confidence = np.max(conf, axis=1)

    print("\nClass mapping: 0=NOW, 1=D1_7, 2=D8_30, 3=GT_30")
    print_horizon_report("Combined Horizon Output From Confidence Scores", y_true_horizon, y_pred_horizon)
    print(f"Mean chosen-class confidence: {float(np.mean(y_pred_confidence)):.4f}")

    cm = confusion_matrix(y_true_horizon, y_pred_horizon, labels=HORIZON_CLASS_LABELS)
    print("Confusion matrix (rows=true, cols=pred, order NOW,D1_7,D8_30,GT_30):")
    print(cm)

    y_pred_policy = predict_horizon_by_policy(model_test_scores, DEFAULT_DEPLOY_THRESHOLDS)
    print_horizon_report("Policy Threshold Horizon Output (Priority NOW->D1_7->D8_30->GT_30)", y_true_horizon, y_pred_policy)

    preview = pd.DataFrame(
        {
            "RUL_DAYS": test_df["RUL_DAYS"].to_numpy(),
            "true_class": y_true_horizon,
            "pred_class": y_pred_horizon,
            "pred_conf": y_pred_confidence,
            "conf_NOW": conf[:, 0],
            "conf_D1_7": conf[:, 1],
            "conf_D8_30": conf[:, 2],
            "conf_GT_30": conf[:, 3],
        }
    )
    print("\nSample confidence outputs (first 10 rows):")
    print(preview.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
