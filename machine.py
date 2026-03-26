"""
Predict equipment failures in advance and identify the most indicative measurement.

Approach:
1. Build a target: whether equipment fails within the next N days (default: 7).
2. Train a time-aware ML classifier (RandomForest with preprocessing).
3. Report predictive metrics on a future holdout period.
4. Use permutation importance to find the strongest measurement signal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
	average_precision_score,
	precision_recall_curve,
	precision_recall_fscore_support,
	roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CSV_PATH = "equipment_failure_data_1.csv"
DEFAULT_HORIZON_DAYS = 7
DEFAULT_TEST_FRACTION = 0.20


@dataclass
class ExperimentResult:
	horizon_days: int
	train_rows: int
	test_rows: int
	train_positive_rate: float
	test_positive_rate: float
	roc_auc: float
	pr_auc: float
	threshold_50_precision: float
	threshold_50_recall: float
	threshold_50_f1: float
	best_f1_threshold: float
	best_f1_precision: float
	best_f1_recall: float
	best_f1_score: float
	most_indicative_measurement: str
	most_indicative_score: float


def _safe_parse_dates(date_series: pd.Series) -> pd.Series:
	"""Parse dates consistently for this dataset format (m/d/yy)."""
	parsed = pd.to_datetime(date_series, format="%m/%d/%y", errors="coerce")
	return parsed


def build_advance_failure_target(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
	"""
	Create label FAIL_WITHIN_HORIZON:
	1 if failure occurs in the next `horizon_days`, excluding current day.
	"""
	out = df.copy()
	out["DATE"] = _safe_parse_dates(out["DATE"])
	out = out.dropna(subset=["DATE"]).sort_values(["ID", "DATE"]).reset_index(drop=True)

	future_fail = out.groupby("ID")["EQUIPMENT_FAILURE"].transform(
		lambda s: s.shift(-1).rolling(horizon_days, min_periods=1).max()
	)
	out["FAIL_WITHIN_HORIZON"] = future_fail.fillna(0).astype(int)

	# Keep only pre-failure rows so we truly predict in advance.
	out = out[out["EQUIPMENT_FAILURE"] == 0].copy()
	return out


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
	"""
	Returns:
	- roc_auc, pr_auc
	- precision/recall/f1 at threshold 0.5
	- best_f1_threshold and best_f1_score from PR curve
	"""
	roc_auc = float(roc_auc_score(y_true, y_score))
	pr_auc = float(average_precision_score(y_true, y_score))

	y_pred_05 = (y_score >= 0.5).astype(int)
	p05, r05, f05, _ = precision_recall_fscore_support(
		y_true, y_pred_05, average="binary", zero_division=0
	)

	precision, recall, thresholds = precision_recall_curve(y_true, y_score)
	f1_curve = (2 * precision * recall) / (precision + recall + 1e-12)

	# precision/recall arrays have one extra element vs thresholds
	# so skip the last element when mapping back to a concrete threshold.
	valid_f1 = f1_curve[:-1] if len(thresholds) > 0 else np.array([f05])
	if len(thresholds) == 0:
		best_threshold = 0.5
		best_f1 = float(f05)
	else:
		best_idx = int(np.nanargmax(valid_f1))
		best_threshold = float(thresholds[best_idx])
		best_f1 = float(valid_f1[best_idx])

	return (
		roc_auc,
		pr_auc,
		float(p05),
		float(r05),
		float(f05),
		best_threshold,
		best_f1,
	)


def run_experiment(csv_path: str, horizon_days: int, test_fraction: float) -> ExperimentResult:
	raw = pd.read_csv(csv_path)

	required_columns = {
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
	missing = required_columns - set(raw.columns)
	if missing:
		raise ValueError(f"Missing expected columns: {sorted(missing)}")

	df = build_advance_failure_target(raw, horizon_days=horizon_days)

	features = [
		"REGION_CLUSTER",
		"MAINTENANCE_VENDOR",
		"MANUFACTURER",
		"WELL_GROUP",
		"AGE_OF_EQUIPMENT",
		"S5",
		"S8",
		"S13",
		"S15",
		"S16",
		"S17",
		"S18",
		"S19",
	]
	sensor_features = [c for c in features if c.startswith("S")]

	cutoff_date = df["DATE"].quantile(1 - test_fraction)
	train_mask = df["DATE"] <= cutoff_date

	train_df = df[train_mask].copy()
	test_df = df[~train_mask].copy()

	X_train = train_df[features]
	y_train = train_df["FAIL_WITHIN_HORIZON"].to_numpy()
	X_test = test_df[features]
	y_test = test_df["FAIL_WITHIN_HORIZON"].to_numpy()

	categorical_cols = ["REGION_CLUSTER", "MAINTENANCE_VENDOR", "MANUFACTURER"]
	numeric_cols = [c for c in features if c not in categorical_cols]

	preprocessor = ColumnTransformer(
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
		n_estimators=300,
		random_state=42,
		class_weight="balanced_subsample",
		n_jobs=-1,
	)

	pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
	pipe.fit(X_train, y_train)

	y_score = pipe.predict_proba(X_test)[:, 1]
	roc_auc, pr_auc, p05, r05, f05, best_threshold, best_f1 = evaluate_predictions(y_test, y_score)

	y_pred_best = (y_score >= best_threshold).astype(int)
	best_p, best_r, _, _ = precision_recall_fscore_support(
		y_test, y_pred_best, average="binary", zero_division=0
	)

	perm = permutation_importance(
		pipe,
		X_test,
		y_test,
		n_repeats=8,
		random_state=42,
		scoring="average_precision",
		n_jobs=-1,
	)
	feature_importance = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
	sensor_importance = feature_importance[sensor_features].sort_values(ascending=False)

	top_sensor = sensor_importance.index[0]
	top_sensor_score = float(sensor_importance.iloc[0])

	return ExperimentResult(
		horizon_days=horizon_days,
		train_rows=len(train_df),
		test_rows=len(test_df),
		train_positive_rate=float(y_train.mean()),
		test_positive_rate=float(y_test.mean()),
		roc_auc=roc_auc,
		pr_auc=pr_auc,
		threshold_50_precision=p05,
		threshold_50_recall=r05,
		threshold_50_f1=f05,
		best_f1_threshold=best_threshold,
		best_f1_precision=float(best_p),
		best_f1_recall=float(best_r),
		best_f1_score=float(best_f1),
		most_indicative_measurement=top_sensor,
		most_indicative_score=top_sensor_score,
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict equipment failure in advance and rank measurements.")
	parser.add_argument("--csv", default=CSV_PATH, help="Path to CSV file")
	parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS, help="Prediction horizon")
	parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION, help="Fraction for time-based test set")
	args = parser.parse_args()

	result = run_experiment(args.csv, args.horizon_days, args.test_fraction)

	print("=== Equipment Failure Prediction (Advance Warning) ===")
	print(f"Prediction horizon: {result.horizon_days} days ahead")
	print(f"Train rows: {result.train_rows:,} | Test rows: {result.test_rows:,}")
	print(
		f"Positive rate (fail-within-horizon): train={result.train_positive_rate:.4f}, "
		f"test={result.test_positive_rate:.4f}"
	)

	print("\nModel performance on future test period:")
	print(f"ROC-AUC: {result.roc_auc:.4f}")
	print(f"PR-AUC:  {result.pr_auc:.4f}")
	print(
		"At threshold 0.50: "
		f"precision={result.threshold_50_precision:.4f}, "
		f"recall={result.threshold_50_recall:.4f}, "
		f"f1={result.threshold_50_f1:.4f}"
	)
	print(
		"At best-F1 threshold: "
		f"threshold={result.best_f1_threshold:.4f}, "
		f"precision={result.best_f1_precision:.4f}, "
		f"recall={result.best_f1_recall:.4f}, "
		f"f1={result.best_f1_score:.4f}"
	)

	print("\nQuestion answers:")
	predictable = result.roc_auc > 0.60 and result.pr_auc > result.test_positive_rate
	if predictable:
		print("1) Can failures be predicted in advance? YES (better than random baseline).")
	else:
		print("1) Can failures be predicted in advance? PARTIALLY (signal is weak).")
	print(
		"2) Most indicative measurement of impending failure: "
		f"{result.most_indicative_measurement} "
		f"(permutation importance={result.most_indicative_score:.5f})."
	)


if __name__ == "__main__":
	main()
