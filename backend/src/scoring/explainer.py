import os

import joblib
import numpy as np
import pandas as pd
import shap

from src.preprocessing.clean import clean_dataset
from src.features.engineer import engineer_features

os.makedirs("saved_models", exist_ok=True)

_lgbm_model = joblib.load("models/lgbm_model.pkl")
_imputer = joblib.load("models/imputer.pkl")
_scaler = joblib.load("models/scaler.pkl")
_feature_names = joblib.load("models/selected_features.pkl")
_risk_scores = pd.read_csv("saved_models/risk_scores.csv").set_index("account_index")
_typology_labels = joblib.load("saved_models/typology_labels.pkl")


_data_path = "data/DataSet.csv"


def _build_processed_matrix():
    raw = pd.read_csv(_data_path)
    if "F3924" not in raw.columns:
        raw = raw.copy()
        raw["F3924"] = 0
    cleaned_df, _ = clean_dataset(raw)
    engineered_df = engineer_features(cleaned_df)

    X_full = engineered_df.drop(columns=["F3924"])
    X_full_imputed = _imputer.transform(X_full)
    X_full_scaled = _scaler.transform(X_full_imputed)
    X_full_scaled = pd.DataFrame(X_full_scaled, columns=X_full.columns, index=X_full.index)
    return X_full_scaled[_feature_names]


def _compute_shap(model, X):
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X)
    if isinstance(raw_shap, list):
        return raw_shap[1]
    elif raw_shap.ndim == 3:
        return raw_shap[:, :, 1]
    return raw_shap


X_processed = _build_processed_matrix()
shap_values = _compute_shap(_lgbm_model, X_processed)


def reload_state(data_path="data/DataSet.csv"):
    global _lgbm_model, _imputer, _scaler, _feature_names, _risk_scores, _typology_labels
    global X_processed, shap_values, _data_path

    _data_path = data_path
    _lgbm_model = joblib.load("models/lgbm_model.pkl")
    _imputer = joblib.load("models/imputer.pkl")
    _scaler = joblib.load("models/scaler.pkl")
    _feature_names = joblib.load("models/selected_features.pkl")
    _risk_scores = pd.read_csv("saved_models/risk_scores.csv").set_index("account_index")
    _typology_labels = joblib.load("saved_models/typology_labels.pkl")

    X_processed = _build_processed_matrix()
    shap_values = _compute_shap(_lgbm_model, X_processed)


def generate_report(account_idx):
    row_pos = X_processed.index.get_loc(account_idx)

    risk_row = _risk_scores.loc[account_idx]
    risk_score = float(risk_row["risk_score"])
    risk_tier = str(risk_row["risk_tier"])
    ml_proba = float(risk_row["ml_score"]) / 100
    true_label = int(risk_row["true_label"])
    typology = str(_typology_labels.loc[account_idx])

    sv = shap_values[row_pos]
    account_values = X_processed.iloc[row_pos]

    order = np.argsort(-np.abs(sv))[:5]
    top_risk_factors = []
    for i in order:
        feature = _feature_names[i]
        shap_val = float(sv[i])
        top_risk_factors.append({
            "feature": feature,
            "shap_value": shap_val,
            "account_value": float(account_values.iloc[i]),
            "direction": "increases risk" if shap_val > 0 else "decreases risk",
        })

    drivers = []
    for f in top_risk_factors[:3]:
        if f["direction"] == "increases risk":
            drivers.append(f"{f['feature']} strongly increases risk")
        else:
            drivers.append(f"{f['feature']} decreases risk")

    investigation_summary = (
        f"This account shows {risk_tier.upper()} risk (score: {risk_score:.1f}). "
        f"Typology: {typology}. "
        f"Key drivers: {', '.join(drivers)}."
    )

    return {
        "account_index": account_idx,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "typology": typology,
        "ml_fraud_probability": ml_proba,
        "true_label": true_label,
        "top_risk_factors": top_risk_factors,
        "investigation_summary": investigation_summary,
    }


def print_report(report):
    print("\n" + "=" * 60)
    print(f"Account {report['account_index']}")
    print("=" * 60)
    print(f"Risk score: {report['risk_score']:.2f}  ({report['risk_tier']})")
    print(f"Typology: {report['typology']}")
    print(f"ML fraud probability: {report['ml_fraud_probability']:.4f}")
    print(f"True label: {report['true_label']}")
    print("Top risk factors:")
    for f in report["top_risk_factors"]:
        print(f"  {f['feature']}: shap={f['shap_value']:.4f}, "
              f"value={f['account_value']:.4f} ({f['direction']})")
    print(f"\nSummary: {report['investigation_summary']}")


if __name__ == "__main__":
    joblib.dump(shap_values, "saved_models/shap_values.pkl")
    print("Saved saved_models/shap_values.pkl")

    sorted_scores = _risk_scores.sort_values("risk_score", ascending=False)

    print("\n\n" + "#" * 60)
    print("TOP 5 HIGHEST RISK ACCOUNTS")
    print("#" * 60)
    for account_idx in sorted_scores.head(5).index:
        print_report(generate_report(account_idx))

    print("\n\n" + "#" * 60)
    print("TOP 3 HIGHEST RISK FALSE POSITIVES (high score, actually legit)")
    print("#" * 60)
    false_positives = sorted_scores[sorted_scores["true_label"] == 0]
    for account_idx in false_positives.head(3).index:
        print_report(generate_report(account_idx))

    print("\n\n" + "#" * 60)
    print("FRAUD ACCOUNTS THAT SCORED LOW RISK (false negatives)")
    print("#" * 60)
    false_negatives = _risk_scores[
        (_risk_scores["true_label"] == 1) & (_risk_scores["risk_tier"] == "Low")
    ]
    if len(false_negatives) == 0:
        print("None found - all fraud accounts scored Medium/High/Critical.")
    else:
        for account_idx in false_negatives.head(2).index:
            print_report(generate_report(account_idx))
