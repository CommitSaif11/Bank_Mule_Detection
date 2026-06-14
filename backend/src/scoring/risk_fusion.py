import os

import joblib
import pandas as pd

from src.preprocessing.clean import clean_dataset
from src.features.engineer import engineer_features

os.makedirs("saved_models", exist_ok=True)

TYPOLOGY_BOOST = {
    "Complicit Mule": 8,
    "Recruited Mule": 5,
    "Exploited Mule": 3,
    "Low Risk": 0,
}


def assign_tier(score):
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Medium"
    elif score <= 80:
        return "High"
    else:
        return "Critical"


if __name__ == "__main__":
    lgbm_model = joblib.load("models/lgbm_model.pkl")
    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/selected_features.pkl")
    anomaly_scores = joblib.load("saved_models/anomaly_scores.pkl")
    typology_labels = joblib.load("saved_models/typology_labels.pkl")

    raw = pd.read_csv("data/DataSet.csv")
    cleaned_df, _ = clean_dataset(raw)
    engineered_df = engineer_features(cleaned_df)

    y_full = engineered_df["F3924"]
    X_full = engineered_df.drop(columns=["F3924"])

    X_full_imputed = imputer.transform(X_full)
    X_full_scaled = scaler.transform(X_full_imputed)
    X_full_scaled = pd.DataFrame(X_full_scaled, columns=X_full.columns, index=X_full.index)
    X_full_selected = X_full_scaled[feature_names]

    ml_proba = lgbm_model.predict_proba(X_full_selected)[:, 1]
    ml_score = pd.Series(ml_proba * 100, index=X_full_selected.index, name="ml_score")

    anomaly_score = anomaly_scores.reindex(X_full_selected.index)
    typology_label = typology_labels.reindex(X_full_selected.index)
    typology_boost = typology_label.map(TYPOLOGY_BOOST)

    risk_score = (
        ml_score * 0.75
        + anomaly_score * 0.15
        + typology_boost * 0.10 * (100 / 8)
    )
    risk_score = risk_score.clip(0, 100)
    risk_tier = risk_score.apply(assign_tier)

    results = pd.DataFrame({
        "account_index": X_full_selected.index,
        "ml_score": ml_score,
        "anomaly_score": anomaly_score,
        "typology_label": typology_label,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "true_label": y_full,
    })

    print("\n" + "=" * 60)
    print("Risk tier distribution (all accounts)")
    print("=" * 60)
    print(results["risk_tier"].value_counts())

    print("\n" + "=" * 60)
    print("Risk tier distribution (fraud accounts only)")
    print("=" * 60)
    fraud_results = results[results["true_label"] == 1]
    print(fraud_results["risk_tier"].value_counts())

    print("\n" + "=" * 60)
    print("Mean risk score")
    print("=" * 60)
    print(f"Fraud accounts: {fraud_results['risk_score'].mean():.4f}")
    print(f"Legit accounts: {results[results['true_label'] == 0]['risk_score'].mean():.4f}")

    n_critical_high = fraud_results["risk_tier"].isin(["Critical", "High"]).sum()
    print(f"\nFraud accounts scored Critical or High: {n_critical_high} / {len(fraud_results)}")

    results.to_csv("saved_models/risk_scores.csv", index=False)
    print("\nSaved saved_models/risk_scores.csv")
