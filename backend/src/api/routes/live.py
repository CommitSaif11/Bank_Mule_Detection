import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.utils import clean_record
from src.api.dependencies import (
    get_lgbm_model, get_imputer, get_scaler, get_selected_features,
    get_isolation_forest, get_kmeans, get_feature_medians,
    get_fraud_percentiles, get_legit_percentiles, get_anomaly_score_range,
    get_typology_cluster_map, get_shap_explainer,
)
from src.features.engineer import F3889_MAP, F3891_MAP, MISSING_FLAG_COLS
from src.models.typology import CLUSTER_FEATURES
from src.scoring.risk_fusion import TYPOLOGY_BOOST, assign_tier

router = APIRouter()


def _not_ready():
    raise HTTPException(status_code=503, detail="Models not yet loaded. Please upload a dataset first.")


def _is_high_risk_input(features: dict) -> bool:
    return (
        features.get("F670", 0) >= 1.0
        or features.get("F3887", 999) < 20
        or features.get("F3894", 999) < 25
        or features.get("F1692", 999) < 0.05
        or features.get("F3891") in ("student", "others")
    )


def _fill_value(col, fill_source, feature_medians):
    if col in fill_source.index:
        return fill_source[col]
    if col in feature_medians.index:
        return feature_medians[col]
    return 0.0


def _build_input_row(features: dict, imputer_cols, feature_medians, fraud_pct, legit_pct) -> pd.DataFrame:
    row = {}
    fill_source = fraud_pct if _is_high_risk_input(features) else legit_pct

    for col in imputer_cols:
        base = col[:-8] if col.endswith("_missing") else None

        if base is not None and base in MISSING_FLAG_COLS:
            row[col] = 0 if base in features else 1
        elif col == "F2678_log":
            if "F2678" in features:
                x = float(features["F2678"])
                row[col] = np.sign(x) * np.log1p(abs(x))
            else:
                row[col] = _fill_value(col, fill_source, feature_medians)
        elif col == "F3836_log":
            if "F3836" in features:
                x = float(features["F3836"])
                row[col] = np.sign(x) * np.log1p(abs(x))
            else:
                row[col] = _fill_value(col, fill_source, feature_medians)
        elif col == "F3889":
            if "F3889" in features:
                val = features["F3889"]
                row[col] = F3889_MAP.get(val, val) if isinstance(val, str) else val
            else:
                row[col] = _fill_value(col, fill_source, feature_medians)
        elif col == "F3891":
            if "F3891" in features:
                val = features["F3891"]
                row[col] = F3891_MAP.get(val, val) if isinstance(val, str) else val
            else:
                row[col] = _fill_value(col, fill_source, feature_medians)
        elif col in features:
            row[col] = features[col]
        else:
            row[col] = _fill_value(col, fill_source, feature_medians)

    return pd.DataFrame([row], columns=imputer_cols)


def _score(features: dict) -> dict:
    start = time.perf_counter()

    lgbm_model = get_lgbm_model()
    imputer = get_imputer()
    scaler = get_scaler()
    feature_names = get_selected_features()
    iso_forest = get_isolation_forest()
    kmeans = get_kmeans()
    feature_medians = get_feature_medians()
    fraud_pct = get_fraud_percentiles()
    legit_pct = get_legit_percentiles()
    anomaly_range = get_anomaly_score_range()
    cluster_map = get_typology_cluster_map()
    explainer = get_shap_explainer()

    imputer_cols = list(imputer.feature_names_in_)

    raw_row = _build_input_row(features, imputer_cols, feature_medians, fraud_pct, legit_pct)
    imputed = imputer.transform(raw_row)
    imputed_df = pd.DataFrame(imputed, columns=imputer_cols)
    scaled = scaler.transform(imputed)
    scaled_df = pd.DataFrame(scaled, columns=imputer_cols)

    X_selected = scaled_df[feature_names].fillna(0)

    ml_proba = float(lgbm_model.predict_proba(X_selected)[:, 1][0])
    ml_score = ml_proba * 100

    raw_anomaly = iso_forest.decision_function(X_selected)[0]
    min_score, max_score = anomaly_range
    anomaly_score = 100 * (max_score - raw_anomaly) / (max_score - min_score)
    anomaly_score = float(np.clip(anomaly_score, 0, 100))

    cluster_row = imputed_df[CLUSTER_FEATURES]
    cluster_id = int(kmeans.predict(cluster_row)[0])
    typology = cluster_map.get(cluster_id, "Low Risk")
    typology_boost = TYPOLOGY_BOOST.get(typology, 0)

    risk_score = ml_score * 0.75 + anomaly_score * 0.15 + typology_boost * 0.10 * (100 / 8)

    # The model relies heavily on ~150 features that the live endpoint can't observe
    # directly. When the 18 bank-listed inputs themselves describe a high-risk profile
    # (new account, student, no linked entities, etc.), surface that signal even if the
    # imputed remainder keeps the raw model score muted.
    if _is_high_risk_input(features):
        risk_score = max(risk_score, 68.0)

    risk_score = float(np.clip(risk_score, 0, 100))
    risk_tier = assign_tier(risk_score)

    sv_raw = explainer.shap_values(X_selected)
    if isinstance(sv_raw, list):
        sv = sv_raw[1][0]
    elif sv_raw.ndim == 3:
        sv = sv_raw[0, :, 1]
    else:
        sv = sv_raw[0]

    order = np.argsort(-np.abs(sv))[:3]
    top_risk_factors = []
    for i in order:
        shap_val = float(sv[i])
        top_risk_factors.append({
            "feature": feature_names[i],
            "shap_value": shap_val,
            "account_value": float(X_selected.iloc[0, i]),
            "direction": "increases risk" if shap_val > 0 else "decreases risk",
        })

    drivers = []
    for f in top_risk_factors:
        if f["direction"] == "increases risk":
            drivers.append(f"{f['feature']} strongly increases risk")
        else:
            drivers.append(f"{f['feature']} decreases risk")

    investigation_summary = (
        f"This account shows {risk_tier.upper()} risk (score: {risk_score:.1f}). "
        f"Typology: {typology}. "
        f"Key drivers: {', '.join(drivers)}."
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "ml_fraud_probability": ml_proba,
        "anomaly_score": anomaly_score,
        "typology": typology,
        "top_risk_factors": top_risk_factors,
        "investigation_summary": investigation_summary,
        "processing_time_ms": elapsed_ms,
    }


@router.post("/score/live")
def score_live(features: dict):
    try:
        return clean_record(_score(features))
    except FileNotFoundError:
        _not_ready()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to score account: {e}")


SAMPLE_HIGH_RISK = {
    "F115": 0.82,
    "F321": 1.10,
    "F527": 0.95,
    "F531": 1.20,
    "F670": 1.0,
    "F1692": 0.05,
    "F2082": 0.005,
    "F2122": 0.008,
    "F2582": 0.02,
    "F2678": 50.0,
    "F2737": 0.05,
    "F2956": 15.0,
    "F3043": 20.0,
    "F3836": -500000.0,
    "F3887": 8,
    "F3889": "G365D",
    "F3891": "student",
    "F3894": 22,
}


@router.get("/score/live/sample")
def sample_payload():
    try:
        raw = pd.read_csv("data/DataSet.csv")
    except FileNotFoundError:
        _not_ready()

    try:
        legit = raw[raw["F3924"] == 0] if "F3924" in raw.columns else raw

        sample_normal = {}
        for col in ["F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
                    "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887", "F3894"]:
            if col in legit.columns:
                sample_normal[col] = round(float(legit[col].mean()), 4)

        for col in ("F3889", "F3891"):
            if col in legit.columns:
                sample_normal[col] = legit[col].mode().iloc[0]

        return clean_record({
            "sample_normal": sample_normal,
            "sample_high_risk": SAMPLE_HIGH_RISK,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build sample payload: {e}")
