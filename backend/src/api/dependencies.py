from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

BASE = Path(__file__).parent.parent.parent  # backend/


@lru_cache(maxsize=1)
def get_risk_scores():
    path = BASE / 'saved_models' / 'risk_scores.csv'
    if not path.exists():
        raise FileNotFoundError("risk_scores.csv not found. Run pipeline first.")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def get_lgbm_model():
    path = BASE / 'models' / 'lgbm_model.pkl'
    if not path.exists():
        raise FileNotFoundError("lgbm_model.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_imputer():
    path = BASE / 'models' / 'imputer.pkl'
    if not path.exists():
        raise FileNotFoundError("imputer.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_scaler():
    path = BASE / 'models' / 'scaler.pkl'
    if not path.exists():
        raise FileNotFoundError("scaler.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_selected_features():
    path = BASE / 'models' / 'selected_features.pkl'
    if not path.exists():
        raise FileNotFoundError("selected_features.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_shap_values():
    path = BASE / 'saved_models' / 'shap_values.pkl'
    if not path.exists():
        raise FileNotFoundError("shap_values.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_typology_labels():
    path = BASE / 'saved_models' / 'typology_labels.pkl'
    if not path.exists():
        raise FileNotFoundError("typology_labels.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_anomaly_scores():
    path = BASE / 'saved_models' / 'anomaly_scores.pkl'
    if not path.exists():
        raise FileNotFoundError("anomaly_scores.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_feature_medians():
    path = BASE / 'saved_models' / 'feature_medians.pkl'
    if not path.exists():
        raise FileNotFoundError("feature_medians.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_fraud_percentiles():
    path = BASE / 'saved_models' / 'fraud_percentiles.pkl'
    if not path.exists():
        return get_feature_medians()
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_legit_percentiles():
    path = BASE / 'saved_models' / 'legit_percentiles.pkl'
    if not path.exists():
        return get_feature_medians()
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_isolation_forest():
    path = BASE / 'saved_models' / 'isolation_forest.pkl'
    if not path.exists():
        raise FileNotFoundError("isolation_forest.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_kmeans():
    path = BASE / 'saved_models' / 'kmeans_typology.pkl'
    if not path.exists():
        raise FileNotFoundError("kmeans_typology.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_anomaly_score_range():
    path = BASE / 'saved_models' / 'anomaly_score_range.pkl'
    if not path.exists():
        return (-0.5, 0.5)
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_typology_cluster_map():
    path = BASE / 'saved_models' / 'typology_cluster_map.pkl'
    if not path.exists():
        return {0: "Complicit Mule", 1: "Recruited Mule", 2: "Exploited Mule", 3: "Low Risk"}
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_shap_explainer():
    return shap.TreeExplainer(get_lgbm_model())


@lru_cache(maxsize=1)
def get_X_processed():
    path = BASE / 'saved_models' / 'X_processed.pkl'
    if not path.exists():
        raise FileNotFoundError("X_processed.pkl not found. Run pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_explainer_state():
    risk_scores = get_risk_scores().set_index("account_index")
    typology_labels = get_typology_labels()
    feature_names = get_selected_features()
    X_processed = get_X_processed()
    shap_values = get_shap_values()

    return {
        "X_processed": X_processed,
        "shap_values": shap_values,
        "risk_scores": risk_scores,
        "typology_labels": typology_labels,
        "feature_names": feature_names,
    }


def clear_cache():
    """Call this after pipeline reruns to reload fresh models."""
    get_risk_scores.cache_clear()
    get_lgbm_model.cache_clear()
    get_imputer.cache_clear()
    get_scaler.cache_clear()
    get_selected_features.cache_clear()
    get_shap_values.cache_clear()
    get_typology_labels.cache_clear()
    get_anomaly_scores.cache_clear()
    get_feature_medians.cache_clear()
    get_fraud_percentiles.cache_clear()
    get_legit_percentiles.cache_clear()
    get_isolation_forest.cache_clear()
    get_kmeans.cache_clear()
    get_anomaly_score_range.cache_clear()
    get_typology_cluster_map.cache_clear()
    get_shap_explainer.cache_clear()
    get_X_processed.cache_clear()
    get_explainer_state.cache_clear()
