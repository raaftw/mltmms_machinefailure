import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA


df = pd.read_csv("equipment_failure_data_1.csv")

target = "EQUIPMENT_FAILURE"

X = df.drop(columns=[target, "ID", "DATE"])
y = df[target]

categorical_features = [
    "REGION_CLUSTER",
    "MAINTENANCE_VENDOR",
    "MANUFACTURER",
    "WELL_GROUP"]

numerical_features = [
    "S15","S17","S13","S16","S19","S18","S8",
    "AGE_OF_EQUIPMENT"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)])

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# transform training data with your existing preprocessor
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train)
print("Before SMOTE:", X_train_processed.shape)
print("After SMOTE:", X_train_smote.shape)

# PCA
pca = PCA(n_components=0.95, random_state=42)  # keep 95% variance

X_train_pca = pca.fit_transform(X_train_smote)
X_val_processed = preprocessor.transform(X_val)
X_val_pca = pca.transform(X_val_processed)

X_test_pca = pca.transform(X_test_processed)

print("Before PCA:", X_train_smote.shape)
print("After PCA:", X_train_pca.shape)

model = LogisticRegression(class_weight="balanced", max_iter=1000)

model.fit(X_train_pca, y_train_smote)
print("LR trained")

# Use validation?
y_val_pred = model.predict(X_val_pca)
y_val_proba = model.predict_proba(X_val_pca)[:, 1]

print("Validation F1:", f1_score(y_val, y_val_pred))
print("Validation ROC-AUC:", roc_auc_score(y_val, y_val_proba))
print(classification_report(y_val, y_val_pred))

models = {
    "LogReg": LogisticRegression(class_weight="balanced", max_iter=1000),
    "RF": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True),
    "XGB": XGBClassifier(eval_metric="logloss")
}

best_f1 = -1
for name, model in models.items():
    model.fit(X_train_pca, y_train_smote)
    y_val_pred = model.predict(X_val_pca)
    y_val_proba = model.predict_proba(X_val_pca)[:, 1]

    print(f"\n{name}")
    f1 = f1_score(y_val, y_val_pred)
    print("F1:", f1)
    print("ROC-AUC:", roc_auc_score(y_val, y_val_proba))
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# Test!
y_test_pred = best_model.predict(X_test_pca)
y_test_proba = best_model.predict_proba(X_test_pca)[:, 1]

print("Model tested: ", best_model_name)
print("Test F1:", f1_score(y_test, y_test_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_test_proba))


plt.figure()
fpr, tpr, _ = roc_curve(y_val, y_val_proba)
plt.plot(fpr, tpr, label="Validation ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()

plt.figure()
precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.show()
