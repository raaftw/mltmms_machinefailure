"""
Time-split LSTM Neural Network tuning for one-vs-rest RUL bins.

This script tunes an LSTM-based model per RUL bin following best practices from
RUL/failure prediction literature (e.g., deep learning for remaining useful life).

Key design choices aligned with papers:
- LSTM for sequence modeling (temporal dependencies in sensor readings)
- Batch normalization for training stability
- Dropout for regularization
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping on validation metric
- Per-bin hyperparameter tuning
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
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
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
except ModuleNotFoundError:
    tf = None
    keras = None
    layers = None
    regularizers = None


# ================================
# User-tunable top-level config
# ================================
CSV_PATH = "equipment_failure_data_1.csv"
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_MAX_TRAIN_ROWS = 0

# Bin cutoffs for one-vs-rest tasks.
BIN_MAX_DAYS = (30,)

# Threshold tuning metric.
GLOBAL_BETA = 2.0
USE_TASK_SPECIFIC_BETA = True
TASK_BETA = {
    "LE_0_DAYS": 3.0,
    "LE_7_DAYS": 1.5,
    "LE_30_DAYS": 0.5,
}

# Selection metric for tuning: "pr_auc" or "fbeta".
SELECTION_METRIC = "pr_auc"
SELECTION_METRIC_CHOICES = ("pr_auc", "fbeta")

# Same feature settings as time_split_binary_horizons.py
DEFAULT_SENSOR_COLS = ("S5", "S8", "S13", "S15", "S16", "S17", "S18", "S19")
DEFAULT_ROLLING_WINDOW = 7
DEFAULT_ROLLING_MIN_PERIODS = 2

# Optional PCA stage.
USE_PCA = True
PCA_N_COMPONENTS = 0.95

# ========================================================
# LSTM hyperparameter candidates per bin.
# Tune these to trade off between capacity and overfitting.
# ========================================================
NN_PARAM_GRID = [
    {
        "lstm_units": 128,
        "lstm_layers": 2,
        "dropout_after_lstm": 0.2,
        "dense_units": 16,
        "l2_reg": 0.0003,
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 120,
    }
]

# LSTM-specific training settings (shared across all configs).
SEQUENCE_LENGTH = 15  # Look-back window (10 time steps).
EARLY_STOPPING_PATIENCE = 12
REDUCE_LR_PATIENCE = 8
REDUCE_LR_FACTOR = 0.5
# Keras training verbosity: 0=silent, 1=progress bar, 2=one line per epoch.
TRAIN_FIT_VERBOSE = 1

# Stage A (fast screening): random-search small budget with short runs.
STAGE_A_NUM_SAMPLES = 12
STAGE_A_EPOCHS = 20
STAGE_A_EARLY_STOPPING_PATIENCE = 4
STAGE_A_REDUCE_LR_PATIENCE = 2
STAGE_A_SEARCH_SPACE = {
    "lstm_units": [64, 128],
    "lstm_layers": [1, 2],
    "dropout_after_lstm": [0.2, 0.3, 0.4],
    "dense_units": [16, 32, 64],
    "l2_reg": [0.0001, 0.0003, 0.001],
    "batch_size": [16, 32, 64],
    "learning_rate": [0.0001, 0.0003, 0.0005, 0.001],
}

# Learning curve plotting.
PLOT_LEARNING_CURVES = True
LEARNING_CURVE_OUTPUT_DIR = "plots/nn_learning_curves"
SHOW_LEARNING_CURVES = False

# Append-only experiment log.
SAVE_EXPERIMENT_LOG = True
EXPERIMENT_LOG_PATH = "experiments/nn_tuning_results.csv"

# Save best model artifacts.
SAVE_BEST_MODEL = True
MODEL_OUTPUT_DIR = "models/nn_best"

RANDOM_STATE = 42

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


def _check_tf() -> None:
    if tf is None or keras is None:
        raise ModuleNotFoundError(
            "TensorFlow is required for NN tuning. Install with:\n"
            "  pip install tensorflow\n"
            "(This may take a few minutes on first install)"
        )

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detected: {len(gpus)} GPU(s) available")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU config warning: {e}")
    else:
        print("No GPU detected; using CPU")


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


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Build sequences for LSTM from time-ordered samples.

    Assumes samples are already sorted by time. For each valid starting point,
    collects seq_len consecutive samples as a sequence.
    """
    X_seq = []
    y_seq = []
    n = len(X)

    for i in range(n - seq_len + 1):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])  # Target is last label in sequence

    return np.array(X_seq), np.array(y_seq)


def build_nn_model(
    input_shape: tuple[int, int],
    lstm_units: int,
    lstm_layers: int,
    dropout_after_lstm: float,
    dense_units: int,
    l2_reg: float,
):
    """Build LSTM model with batch norm and dropout for RUL prediction.

    Input shape: (sequence_length, n_features)
    """
    model = keras.Sequential()

    model.add(layers.Input(shape=input_shape))

    model.add(
        layers.LSTM(
            lstm_units,
            return_sequences=(lstm_layers > 1),
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )
    model.add(layers.BatchNormalization())

    for layer_idx in range(lstm_layers - 1):
        model.add(layers.Dropout(dropout_after_lstm))
        model.add(
            layers.LSTM(
                lstm_units,
                return_sequences=(layer_idx < lstm_layers - 2),
                kernel_regularizer=regularizers.l2(l2_reg),
            )
        )
        model.add(layers.BatchNormalization())

    model.add(layers.Dropout(dropout_after_lstm))
    model.add(layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_after_lstm))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


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


def _build_stage_a_param_grid(num_samples: int, epochs: int, seed: int) -> list[dict]:
    """Sample a small random config set from a compact, high-impact search space."""
    if num_samples <= 0:
        raise ValueError("--stage-a-samples must be > 0")
    if epochs <= 0:
        raise ValueError("--stage-a-epochs must be > 0")

    rng = np.random.default_rng(seed)

    all_candidates: list[dict] = []
    for lstm_units in STAGE_A_SEARCH_SPACE["lstm_units"]:
        for lstm_layers in STAGE_A_SEARCH_SPACE["lstm_layers"]:
            for dropout_after_lstm in STAGE_A_SEARCH_SPACE["dropout_after_lstm"]:
                for dense_units in STAGE_A_SEARCH_SPACE["dense_units"]:
                    for l2_reg in STAGE_A_SEARCH_SPACE["l2_reg"]:
                        for batch_size in STAGE_A_SEARCH_SPACE["batch_size"]:
                            for learning_rate in STAGE_A_SEARCH_SPACE["learning_rate"]:
                                all_candidates.append(
                                    {
                                        "lstm_units": lstm_units,
                                        "lstm_layers": lstm_layers,
                                        "dropout_after_lstm": dropout_after_lstm,
                                        "dense_units": dense_units,
                                        "l2_reg": l2_reg,
                                        "batch_size": batch_size,
                                        "learning_rate": learning_rate,
                                        "epochs": epochs,
                                    }
                                )

    take = min(num_samples, len(all_candidates))
    selected_idx = rng.choice(len(all_candidates), size=take, replace=False)
    return [all_candidates[int(i)] for i in selected_idx]


def save_learning_curve(
    history,
    task_name: str,
    config_id: int,
    output_dir: str,
    show_plot: bool,
) -> None:
    """Save train/validation learning curves across epochs."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for learning-curve plotting. Install with `pip install matplotlib`."
        ) from exc

    hist = history.history
    epochs = np.arange(1, len(hist.get("loss", [])) + 1)
    if len(epochs) == 0:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, hist.get("loss", []), label="train_loss")
    axes[0].plot(epochs, hist.get("val_loss", []), label="val_loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    has_pr_curve = False
    if "pr_auc" in hist:
        axes[1].plot(epochs, hist.get("pr_auc", []), label="train_pr_auc")
        has_pr_curve = True
    if "val_pr_auc" in hist:
        axes[1].plot(epochs, hist.get("val_pr_auc", []), label="val_pr_auc")
        has_pr_curve = True
    if not has_pr_curve:
        axes[1].text(0.5, 0.5, "PR-AUC history unavailable", ha="center", va="center")
    axes[1].set_title("PR-AUC vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PR-AUC")
    axes[1].grid(alpha=0.3)
    if has_pr_curve:
        axes[1].legend()

    fig.suptitle(f"Learning Curves - {task_name} - config_{config_id:02d}")
    out_path = out_dir / f"learning_curve_{task_name.lower()}_config_{config_id:02d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)


def append_experiment_log(
    out_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    args,
    task_definitions: list[tuple[str, int]],
    effective_param_grid: list[dict],
    effective_early_stopping_patience: int,
    effective_reduce_lr_patience: int,
    stage_a_mode: bool,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    train_seq_rows: int,
    val_seq_rows: int,
    test_seq_rows: int,
) -> tuple[str, str]:
    """Append settings and all per-config results for reproducibility."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logged_at_utc = datetime.now(timezone.utc).isoformat()

    bins_only = [max_days for _, max_days in task_definitions]
    settings_snapshot = {
        "csv": args.csv,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "max_train_rows": args.max_train_rows,
        "metric": args.metric,
        "beta": args.beta,
        "sequence_length": args.sequence_length,
        "bin_max_days": bins_only,
        "use_task_specific_beta": USE_TASK_SPECIFIC_BETA,
        "task_beta": TASK_BETA,
        "use_pca": USE_PCA,
        "pca_n_components": PCA_N_COMPONENTS,
        "rolling_window": DEFAULT_ROLLING_WINDOW,
        "rolling_min_periods": DEFAULT_ROLLING_MIN_PERIODS,
        "early_stopping_patience": effective_early_stopping_patience,
        "reduce_lr_patience": effective_reduce_lr_patience,
        "reduce_lr_factor": REDUCE_LR_FACTOR,
        "train_fit_verbose": TRAIN_FIT_VERBOSE,
        "nn_param_grid": effective_param_grid,
        "stage_a_mode": stage_a_mode,
    }

    records = out_df.copy()
    records["params_json"] = records["params"].apply(lambda p: json.dumps(p, sort_keys=True))
    records = records.drop(columns=["params"])
    records.insert(0, "run_id", run_id)
    records.insert(1, "logged_at_utc", logged_at_utc)

    best_keys = set(zip(summary_df["task"].tolist(), summary_df["config_id"].tolist()))
    records["best_for_task"] = [
        (task, config_id) in best_keys for task, config_id in zip(records["task"], records["config_id"])
    ]

    records["csv_path"] = args.csv
    records["train_fraction"] = args.train_fraction
    records["val_fraction"] = args.val_fraction
    records["max_train_rows"] = args.max_train_rows
    records["metric"] = args.metric
    records["beta"] = args.beta
    records["sequence_length"] = args.sequence_length
    records["bins_json"] = json.dumps(bins_only)
    records["settings_json"] = json.dumps(settings_snapshot, sort_keys=True)

    records["rows_train"] = train_rows
    records["rows_val"] = val_rows
    records["rows_test"] = test_rows
    records["seq_rows_train"] = train_seq_rows
    records["seq_rows_val"] = val_seq_rows
    records["seq_rows_test"] = test_seq_rows

    log_path = Path(EXPERIMENT_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    records.to_csv(log_path, mode="a", header=write_header, index=False)
    return run_id, str(log_path.resolve())


def _set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
        deterministic: If False, skips TF GPU determinism flags (better performance, less stable across runs)
    
    Controls randomness in:
    - Python's random module
    - NumPy
    - TensorFlow/Keras (including GPU operations and dropout)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Enable deterministic GPU operations for reproducible results
    # Disable if --stochastic for better performance at cost of variance
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    else:
        os.environ.pop("TF_DETERMINISTIC_OPS", None)
        os.environ.pop("TF_CUDNN_DETERMINISTIC", None)


def main() -> None:
    _check_tf()

    parser = argparse.ArgumentParser(description="Tune LSTM NN per RUL bin with time split")
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
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument(
        "--stage-a",
        action="store_true",
        help="Run fast Stage A random screening instead of the fixed NN_PARAM_GRID.",
    )
    parser.add_argument("--stage-a-samples", type=int, default=STAGE_A_NUM_SAMPLES)
    parser.add_argument("--stage-a-epochs", type=int, default=STAGE_A_EPOCHS)
    parser.add_argument("--stage-a-early-stopping-patience", type=int, default=STAGE_A_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--stage-a-reduce-lr-patience", type=int, default=STAGE_A_REDUCE_LR_PATIENCE)
    parser.add_argument("--stage-a-seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Disable deterministic GPU ops for better performance (less consistent across runs).",
    )
    args = parser.parse_args()

    # Apply deterministic seeding (disabled with --stochastic for better performance)
    _set_seed(RANDOM_STATE, deterministic=not args.stochastic)

    if args.stochastic:
        print("[INFO] Running in stochastic mode (non-deterministic GPU ops for better performance)")

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    model_run_dir: Path | None = None
    if SAVE_BEST_MODEL:
        model_run_dir = Path(MODEL_OUTPUT_DIR) / run_stamp
        model_run_dir.mkdir(parents=True, exist_ok=True)

    effective_param_grid = NN_PARAM_GRID
    effective_early_stopping_patience = EARLY_STOPPING_PATIENCE
    effective_reduce_lr_patience = REDUCE_LR_PATIENCE

    if args.stage_a:
        effective_param_grid = _build_stage_a_param_grid(
            num_samples=args.stage_a_samples,
            epochs=args.stage_a_epochs,
            seed=args.stage_a_seed,
        )
        effective_early_stopping_patience = args.stage_a_early_stopping_patience
        effective_reduce_lr_patience = args.stage_a_reduce_lr_patience

        if effective_early_stopping_patience <= 0:
            raise ValueError("--stage-a-early-stopping-patience must be > 0")
        if effective_reduce_lr_patience <= 0:
            raise ValueError("--stage-a-reduce-lr-patience must be > 0")

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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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

    X_train_pre = pre.fit_transform(X_train)
    X_val_pre = pre.transform(X_val)
    X_test_pre = pre.transform(X_test)

    if USE_PCA:
        pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
        X_train_pre = pca.fit_transform(X_train_pre)
        X_val_pre = pca.transform(X_val_pre)
        X_test_pre = pca.transform(X_test_pre)

    X_train_seq, y_train_seq_raw = build_sequences(X_train_pre, train_df["RUL_DAYS"].values, args.sequence_length)
    X_val_seq, y_val_seq_raw = build_sequences(X_val_pre, val_df["RUL_DAYS"].values, args.sequence_length)
    X_test_seq, y_test_seq_raw = build_sequences(X_test_pre, test_df["RUL_DAYS"].values, args.sequence_length)

    task_definitions = _make_task_definitions(tuple(args.bin_max_days))
    if len(task_definitions) == 0:
        raise ValueError("At least one bin must be provided in BIN_MAX_DAYS or --bin-max-days")

    print("=== Time-Split NN (LSTM) Bin Tuning ===")
    print(f"Rows (before sequencing): train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Rows (after sequencing): train={len(X_train_seq):,}, val={len(X_val_seq):,}, test={len(X_test_seq):,}")
    print(f"Feature dimension: {X_train_seq.shape[2]}")
    print(f"Metric used to pick best config: {args.metric}")
    print(f"Bins: {[d for _, d in task_definitions]}")
    print(f"NN candidate configs: {len(effective_param_grid)}")
    print(f"PCA enabled: {USE_PCA} (n_components={PCA_N_COMPONENTS})")
    if args.stage_a:
        print(
            "Stage A mode: ON "
            f"(samples={args.stage_a_samples}, epochs={args.stage_a_epochs}, "
            f"early_stop_patience={effective_early_stopping_patience}, "
            f"reduce_lr_patience={effective_reduce_lr_patience})"
        )

    all_rows: list[TuneResult] = []

    for task_name, max_days in task_definitions:
        y_train = (y_train_seq_raw <= max_days).astype(int)
        y_val = (y_val_seq_raw <= max_days).astype(int)
        y_test = (y_test_seq_raw <= max_days).astype(int)

        beta_task = TASK_BETA.get(task_name, args.beta) if USE_TASK_SPECIFIC_BETA else args.beta

        print(f"\nTask {task_name}: RUL <= {max_days}")
        print(
            f"Positive rates: train={float(np.mean(y_train)):.4f}, val={float(np.mean(y_val)):.4f}, test={float(np.mean(y_test)):.4f}"
        )

        best_result: TuneResult | None = None
        best_model = None

        for idx, params in enumerate(effective_param_grid, start=1):
            print(f"  Config {idx:02d}/{len(effective_param_grid)}...", end="", flush=True)

            model = build_nn_model(
                input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                lstm_units=params["lstm_units"],
                lstm_layers=params["lstm_layers"],
                dropout_after_lstm=params["dropout_after_lstm"],
                dense_units=params["dense_units"],
                l2_reg=params["l2_reg"],
            )

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                loss="binary_crossentropy",
                metrics=["accuracy", keras.metrics.AUC(curve="PR", name="pr_auc")],
            )

            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=effective_early_stopping_patience,
                restore_best_weights=True,
            )
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=effective_reduce_lr_patience,
                min_lr=1e-6,
            )

            history = model.fit(
                X_train_seq,
                y_train,
                validation_data=(X_val_seq, y_val),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=[early_stop, reduce_lr],
                verbose=TRAIN_FIT_VERBOSE,
                shuffle=False,  # Deterministic batch ordering
            )

            if PLOT_LEARNING_CURVES:
                save_learning_curve(
                    history=history,
                    task_name=task_name,
                    config_id=idx,
                    output_dir=LEARNING_CURVE_OUTPUT_DIR,
                    show_plot=SHOW_LEARNING_CURVES,
                )

            val_scores = model.predict(X_val_seq, verbose=0).flatten()
            test_scores = model.predict(X_test_seq, verbose=0).flatten()

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

            print(f" val_{args.metric}={val_metric:.4f}, thr={threshold:.4f}, test_fbeta={row.fbeta:.4f}")

            if best_result is None or row.val_metric > best_result.val_metric:
                best_result = row
                best_model = model

        if best_result is None:
            raise RuntimeError(f"No successful NN config for task {task_name}")

        print("  Best config for this task:")
        print(f"    config_id={best_result.config_id} val_{args.metric}={best_result.val_metric:.4f}")
        print(f"    threshold={best_result.threshold:.4f}")
        print(
            f"    test metrics: precision={best_result.precision:.4f} recall={best_result.recall:.4f} "
            f"f1={best_result.f1:.4f} fbeta={best_result.fbeta:.4f} pr_auc={best_result.pr_auc:.4f}"
        )
        print(
            f"    params: lstm_units={best_result.params['lstm_units']}, lstm_layers={best_result.params['lstm_layers']}, "
            f"lr={best_result.params['learning_rate']}, batch_size={best_result.params['batch_size']}"
        )

        if SAVE_BEST_MODEL and best_model is not None and model_run_dir is not None:
            model_path = model_run_dir / f"best_model_{task_name.lower()}.keras"
            metadata_path = model_run_dir / f"best_model_{task_name.lower()}_metadata.json"

            best_model.save(model_path)

            metadata = {
                "task": task_name,
                "run_stamp": run_stamp,
                "selection_metric": args.metric,
                "beta_task": beta_task,
                "threshold": float(best_result.threshold),
                "val_metric": float(best_result.val_metric),
                "params": best_result.params,
                "sequence_length": args.sequence_length,
                "bins": [d for _, d in task_definitions],
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

            print(f"    saved model: {model_path.resolve()}")
            print(f"    saved metadata: {metadata_path.resolve()}")

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

    if SAVE_EXPERIMENT_LOG:
        run_id, log_path = append_experiment_log(
            out_df=out,
            summary_df=summary,
            args=args,
            task_definitions=task_definitions,
            effective_param_grid=effective_param_grid,
            effective_early_stopping_patience=effective_early_stopping_patience,
            effective_reduce_lr_patience=effective_reduce_lr_patience,
            stage_a_mode=args.stage_a,
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
            train_seq_rows=len(X_train_seq),
            val_seq_rows=len(X_val_seq),
            test_seq_rows=len(X_test_seq),
        )
        print(f"\nSaved experiment rows to: {log_path}")
        print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
