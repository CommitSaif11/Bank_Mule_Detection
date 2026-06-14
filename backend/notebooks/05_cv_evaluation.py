import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.preprocessing.clean import clean_dataset
from src.features.engineer import engineer_features

base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
raw = pd.read_csv(os.path.join(base, "data/DataSet.csv"))
cleaned_df, _ = clean_dataset(raw)
engineered_df = engineer_features(cleaned_df)

X = engineered_df.drop(columns=["F3924"])
y = engineered_df["F3924"]

feature_names = joblib.load(os.path.join(base, "models/selected_features.pkl"))
feature_names = [c for c in feature_names if c in X.columns]
X = X[feature_names]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {"LightGBM": [], "XGBoost": [], "RandomForest": []}
fraud_counts = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    fraud_counts.append(int(y_test.sum()))

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_imp, y_train)

    models = {
        "LightGBM": LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=5, random_state=42, n_jobs=-1, verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=5, eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }

    print(f"\n{'=' * 60}")
    print(f"Fold {fold} - fraud cases in test set: {fraud_counts[-1]}")
    print(f"{'=' * 60}")

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_imp)
        y_proba = model.predict_proba(X_test_imp)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)

        recall = report["1"]["recall"]
        precision = report["1"]["precision"]
        f1 = report["1"]["f1-score"]

        results[name].append({
            "recall": recall, "precision": precision, "f1": f1,
            "auc_roc": auc_roc, "auc_pr": auc_pr,
        })

        print(f"{name:<15} Recall={recall:.4f}  Precision={precision:.4f}  F1={f1:.4f}  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}")

print(f"\n{'=' * 60}")
print(f"Fraud cases per fold's test set: {fraud_counts}")

print(f"\n{'=' * 60}")
print("5-Fold CV Summary (mean ± std)")
print(f"{'=' * 60}")
print(f"{'Model':<15} {'Recall':<20} {'Precision':<20} {'F1':<20} {'AUC-ROC':<20} {'AUC-PR':<20}")
for name, fold_results in results.items():
    recalls = [r["recall"] for r in fold_results]
    precisions = [r["precision"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    aucs = [r["auc_roc"] for r in fold_results]
    aucprs = [r["auc_pr"] for r in fold_results]

    print(f"{name:<15} "
          f"{np.mean(recalls):.4f}±{np.std(recalls):.4f}    "
          f"{np.mean(precisions):.4f}±{np.std(precisions):.4f}    "
          f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}    "
          f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}    "
          f"{np.mean(aucprs):.4f}±{np.std(aucprs):.4f}")
