import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.preprocessing.clean import clean_dataset
from src.features.engineer import engineer_features
from src.preprocessing.split_and_impute import prepare_data
from src.features.select_features import select_features

os.makedirs("saved_models", exist_ok=True)


if __name__ == "__main__":
    raw = pd.read_csv("data/DataSet.csv")
    cleaned_df, _ = clean_dataset(raw)
    engineered_df = engineer_features(cleaned_df)
    X_train, X_test, y_train, y_test = prepare_data(engineered_df)
    X_train_selected, X_test_selected, feature_names = select_features(X_train, y_train, X_test)

    # Build full X (all 9082 accounts) imputed + scaled with the saved pipeline
    X_full = engineered_df.drop(columns=["F3924"])
    y_full = engineered_df["F3924"]

    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")

    X_full_imputed = imputer.transform(X_full)
    X_full_scaled = scaler.transform(X_full_imputed)
    X_full_scaled = pd.DataFrame(X_full_scaled, columns=X_full.columns, index=X_full.index)
    X_full_selected = X_full_scaled[feature_names]

    # Train IsolationForest on train set only
    iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_selected)

    # Score all accounts
    raw_scores = iso_forest.decision_function(X_full_selected)

    # Normalize to 0-100, higher = more anomalous
    # decision_function: lower (more negative) = more anomalous, so invert
    min_score, max_score = raw_scores.min(), raw_scores.max()
    anomaly_scores = 100 * (max_score - raw_scores) / (max_score - min_score)
    anomaly_scores = pd.Series(anomaly_scores, index=X_full_selected.index, name="anomaly_score")

    fraud_mask = y_full == 1
    print(f"\nMean anomaly score - fraud accounts: {anomaly_scores[fraud_mask].mean():.4f}")
    print(f"Mean anomaly score - legit accounts: {anomaly_scores[~fraud_mask].mean():.4f}")

    top_5pct_threshold = anomaly_scores.quantile(0.95)
    top_5pct_mask = anomaly_scores >= top_5pct_threshold
    fraud_in_top5pct = (fraud_mask & top_5pct_mask).sum()
    pct_fraud_in_top5pct = 100 * fraud_in_top5pct / fraud_mask.sum()
    print(f"\nTop 5% anomaly threshold: {top_5pct_threshold:.4f}")
    print(f"Fraud accounts in top 5% most anomalous: {fraud_in_top5pct} / {fraud_mask.sum()} "
          f"({pct_fraud_in_top5pct:.2f}%)")

    joblib.dump(iso_forest, "saved_models/isolation_forest.pkl")
    joblib.dump(anomaly_scores, "saved_models/anomaly_scores.pkl")

    print("\nSaved saved_models/isolation_forest.pkl")
    print("Saved saved_models/anomaly_scores.pkl")
