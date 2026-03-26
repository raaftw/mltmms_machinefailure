"""
Compare dimensionality-reduction strategies with a strict time split.

Goal:
- One learning method (LogisticRegression)
- Same train/val/test time split for every run
- Baseline without dimensionality reduction
- Multiple dimensionality-reduction variants

Target:
- IMMINENT_FAILURE = 1 if failure occurs within `--imminent-days`

Run:
  python time_split_dr_experiments.py --csv equipment_failure_data_1.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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


CSV_PATH = "equipment_failure_data_1.csv"
DEFAULT_IMMINENT_DAYS = 7
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_BETA = 2.0
DEFAULT_SVD_COMPONENTS = 40
DEFAULT_PCA_VARIANCE = 0.95
DEFAULT_MAX_TRAIN_ROWS = 0


@dataclass
class ExperimentResult:
    variant: str
    threshold: float
    precision: float
    recall: float
    f1: float
    fbeta: float
    pr_auc: float
    roc_auc: float
    alert_rate: float


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


def add_imminent_target(df: pd.DataFrame, imminent_days: int) -> pd.DataFrame:
    out = df.copy()
    out["IMMINENT_FAILURE"] = (out["RUL_DAYS"] <= imminent_days).astype(int)
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


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, beta: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5

    fbeta = (1 + beta**2) * precision[:-1] * recall[:-1] / ((beta**2 * precision[:-1]) + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(fbeta))])


def evaluate(variant: str, y_true: np.ndarray, y_score: np.ndarray, threshold: float, beta: float) -> ExperimentResult:
    y_pred = (y_score >= threshold).astype(int)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fbeta = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))

    return ExperimentResult(
        variant=variant,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        fbeta=fbeta,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        alert_rate=float(y_pred.mean()),
    )


def make_preprocessor(
    variant: str,
    categorical_cols: list[str],
    numeric_cols: list[str],
    svd_components: int,
    pca_variance: float,
) -> ColumnTransformer:
    cat_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
    num_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]

    if variant in {"svd_cat", "both"}:
        cat_steps.append(("svd", TruncatedSVD(n_components=svd_components, random_state=42)))

    if variant in {"pca_num", "both"}:
        num_steps.append(("pca", PCA(n_components=pca_variance, random_state=42)))

    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=cat_steps), categorical_cols),
            ("num", Pipeline(steps=num_steps), numeric_cols),
        ],
        remainder="drop",
    )


def run_variant(
    variant: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    categorical_cols: list[str],
    numeric_cols: list[str],
    beta: float,
    svd_components: int,
    pca_variance: float,
) -> ExperimentResult:
    pre = make_preprocessor(
        variant=variant,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        svd_components=svd_components,
        pca_variance=pca_variance,
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    val_score = pipe.predict_proba(X_val)[:, 1]
    test_score = pipe.predict_proba(X_test)[:, 1]
    threshold = tune_threshold(y_val, val_score, beta=beta)

    return evaluate(variant, y_test, test_score, threshold=threshold, beta=beta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-split DR comparison with one model")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to CSV")
    parser.add_argument("--imminent-days", type=int, default=DEFAULT_IMMINENT_DAYS)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="F-beta used for threshold tuning")
    parser.add_argument("--svd-components", type=int, default=DEFAULT_SVD_COMPONENTS)
    parser.add_argument("--pca-variance", type=float, default=DEFAULT_PCA_VARIANCE)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=DEFAULT_MAX_TRAIN_ROWS,
        help="If > 0, randomly cap training rows for faster experiments",
    )
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
    data = add_imminent_target(data, imminent_days=args.imminent_days)
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
    y_train = train_df["IMMINENT_FAILURE"].to_numpy(dtype=int)
    X_val = val_df[feature_cols]
    y_val = val_df["IMMINENT_FAILURE"].to_numpy(dtype=int)
    X_test = test_df[feature_cols]
    y_test = test_df["IMMINENT_FAILURE"].to_numpy(dtype=int)

    variants = [
        "baseline",   # no dimensionality reduction
        "pca_num",    # PCA on numeric branch only
        "svd_cat",    # SVD on one-hot categorical branch only
        "both",       # PCA numeric + SVD categorical
    ]

    results: list[ExperimentResult] = []
    for variant in variants:
        print(f"[INFO] Running variant: {variant}", flush=True)
        result = run_variant(
            variant=variant,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            beta=args.beta,
            svd_components=args.svd_components,
            pca_variance=args.pca_variance,
        )
        results.append(result)
        print(
            f"[INFO] Done {variant}: precision={result.precision:.3f}, "
            f"recall={result.recall:.3f}, f_beta={result.fbeta:.3f}",
            flush=True,
        )

    out = pd.DataFrame([r.__dict__ for r in results]).sort_values("fbeta", ascending=False)

    print("=== Time-Split DR Experiment ===")
    print(f"Rows: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    print(
        f"Positive rate: train={y_train.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}"
    )
    print(f"Threshold tuning objective: F-beta with beta={args.beta}")
    print("\nVariants:")
    print("- baseline: no dimensionality reduction")
    print("- pca_num: PCA on numeric features")
    print("- svd_cat: TruncatedSVD on one-hot categorical features")
    print("- both: PCA numeric + SVD categorical")

    print("\nResults on TEST (sorted by tuned F-beta):")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = out.iloc[0]
    print("\nBest variant by tuned F-beta:")
    print(
        f"{best['variant']} | threshold={best['threshold']:.4f} | "
        f"precision={best['precision']:.3f} | recall={best['recall']:.3f} | "
        f"f_beta={best['fbeta']:.3f} | alert_rate={best['alert_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
